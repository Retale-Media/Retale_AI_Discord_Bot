import os
import aiohttp
import io
import json
import asyncio
import subprocess
from openai import OpenAI
from datetime import datetime
import concurrent.futures
from interactions import Client, Intents, OptionType, slash_command, SlashContext, slash_option, File
from interactions.api.voice.audio import AudioVolume  # For playing audio

api_key_file_path ='\\Directory for\\OPENAI_API_KEY.txt'
json_file_path = '\\Directory for\\assistants_config.json'
discord_token_file_path = '\\Directory for\\DISCORD_TOKEN.txt'
user_input_path = '\\Directory for\\user_input'
ai_output_path = '\\Directory for\\ai_output'

os.makedirs(user_input_path, exist_ok=True)
os.makedirs(ai_output_path, exist_ok=True)

# Read the API key from the file
with open(api_key_file_path, 'r') as file:
    openai_api_key = file.read().strip()

# Read the Discord token from the file
with open(discord_token_file_path, 'r') as file:
    discord_token = file.read().strip()

# Initialize OpenAI client with the API key
client = OpenAI(api_key=openai_api_key)

# Global toggle variables
toggle_audio_file = True
toggle_audio_response = True
toggle_text_response = True
toggle_user_transcription = True
toggle_illustrations = False

# Define the intents
intents = Intents.DEFAULT | Intents.GUILD_MESSAGES | Intents.GUILD_VOICE_STATES

# Global variable to keep track of the recording state
is_recording = False

# Global variable to store the last GPT-4 response
last_gpt_response = ""

# Initialize Discord client with the token and intents
bot = Client(token=discord_token, intents=intents)

# Load assistants from the JSON file
with open(json_file_path, 'r') as file:
    assistants = json.load(file)

# Store current assistant_id and thread_id
current_assistant = {"assistant_id": None, "thread_id": None}

# Toggle commands
@slash_command(name="toggle_audio_file", description="Toggle the audio file feature")
async def toggle_audio_file_cmd(ctx):
    global toggle_audio_file
    toggle_audio_file = not toggle_audio_file
    await ctx.send(f"Audio file feature {'enabled' if toggle_audio_file else 'disabled'}.")

@slash_command(name="toggle_audio_response", description="Toggle the audio response feature")
async def toggle_audio_response_cmd(ctx):
    global toggle_audio_response
    toggle_audio_response = not toggle_audio_response
    await ctx.send(f"Audio response feature {'enabled' if toggle_audio_response else 'disabled'}.")

@slash_command(name="toggle_text_response", description="Toggle the text response feature")
async def toggle_text_response_cmd(ctx):
    global toggle_text_response
    toggle_text_response = not toggle_text_response
    await ctx.send(f"Text response feature {'enabled' if toggle_text_response else 'disabled'}.")
    
@slash_command(name="toggle_user_transcription", description="Toggle the user transcript feature")
async def toggle_user_transcription_cmd(ctx):
    global toggle_user_transcription
    toggle_user_transcription = not toggle_user_transcription
    await ctx.send(f"User transcription feature {'enabled' if toggle_user_transcription else 'disabled'}.")
    
@slash_command(name="toggle_illustrations", description="Toggle the illustration feature")
async def toggle_illustrations_cmd(ctx):
    global toggle_illustrations
    toggle_illustrations = not toggle_illustrations
    await ctx.send(f"Illustration feature {'enabled' if toggle_illustrations else 'disabled'}.")

@slash_command(name="select_assistant", description="Select an assistant to use")
@slash_option(
    name="assistant_name", 
    description="Enter the name of the assistant", 
    required=True, 
    opt_type=OptionType.STRING
)

async def select_assistant(ctx, assistant_name: str):
    if assistant_name in assistants:
        current_assistant["assistant_id"] = assistants[assistant_name]["assistant_id"]
        current_assistant["thread_id"] = assistants[assistant_name]["thread_id"]
        await ctx.send(f"Assistant {assistant_name} selected.")
    else:
        await ctx.send("Assistant not found. Please provide a valid assistant name.")

# Slash command to create an image
@slash_command(name="create_image", description="Generate an image from a prompt")
@slash_option(
    name="prompt", 
    description="Enter the prompt for the image", 
    required=True, 
    opt_type=OptionType.STRING
)

async def create_image(ctx, prompt: str):
    if not prompt:
        await ctx.send("Please provide a prompt for the image.")
        return

    await ctx.send("Generating an image, please wait...")

    image_url = await generate_image(prompt)
    if image_url:
        image_data = await download_image(image_url)
        if image_data:
            await ctx.send(file=File(image_data, 'image.png'))
        else:
            await ctx.send("Failed to download the image.")
    else:
        await ctx.send("Failed to generate an image.")
        
@slash_command(name="listen", description="Start recording audio")
async def listen(ctx):
    global is_recording
    if ctx.author.voice:
        voice_state = await ctx.author.voice.channel.connect()
        await voice_state.start_recording()
        await ctx.send("Recording started. Use the /stop command to finish.")
        is_recording = True
    else:
        await ctx.send("You need to be in a voice channel.")

@slash_command(name="stop_listening", description="Stop recording audio and process")
async def stop_listening(ctx):
    global is_recording
    if ctx.author.voice and is_recording:
        voice_state = await ctx.author.voice.channel.connect()
        await voice_state.stop_recording()
        is_recording = False

        await ctx.send("Processing your request, please wait...")
        asyncio.create_task(process_recording(voice_state, ctx))
    else:
        await ctx.send("Either you are not in a voice channel or no recording is in progress.")

async def process_recording(voice_state, ctx):
    audio_files = []

    # Collect individual audio files from the recording
    for user_id, audio_data in voice_state.recorder.output.items():
        user_audio_file_name = os.path.join(user_input_path, f"{user_id}_audio.mp3")
        with open(user_audio_file_name, 'wb') as audio_file:
            audio_file.write(audio_data.getvalue())
        audio_files.append(user_audio_file_name)

    combined_audio_file = os.path.join(user_input_path, "combined_audio.mp3")
    combine_audio_files(audio_files, combined_audio_file)

    transcript = await speech_to_text(client, combined_audio_file)
    if transcript:
        if toggle_user_transcription:
            transcript_messages = [transcript[i:i+1900] for i in range(0, len(transcript), 1900)]
            for msg in transcript_messages:
                await ctx.channel.send(f"**User Input:** {msg}")

        gpt_response = await ask_assistant(client, transcript, current_assistant["assistant_id"], current_assistant["thread_id"])
        gpt_text_response = gpt_response.text if hasattr(gpt_response, 'text') else str(gpt_response)

        update_last_gpt_response(gpt_text_response)

        if toggle_text_response:
            gpt_messages = [gpt_text_response[i:i+1900] for i in range(0, len(gpt_text_response), 1900)]
            for msg in gpt_messages:
                await ctx.channel.send(f"**AI Output:** {msg}")

        if toggle_audio_response:
            # Generate a unique file name using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ai_audio_file_name = f'AI_Output_{timestamp}.mp3'
            ai_audio_file_path = await text_to_speech(client, gpt_text_response, ai_audio_file_name)

            # Play the audio file in the voice channel
            audio_source = AudioVolume(ai_audio_file_path)
            await voice_state.play(audio_source)
            await ctx.channel.send(file=File(ai_audio_file_path))
        
            if toggle_illustrations:
                await generate_and_send_illustration(ctx)
    else:
        await ctx.channel.send("No transcript available.")

    await voice_state.disconnect()

def combine_audio_files(files, output_file):
    command = ["ffmpeg", "-y"]
    for file in files:
        command.extend(["-i", file])
    output_path = os.path.join(user_input_path, output_file)
    command.extend(["-filter_complex", f"amix=inputs={len(files)}:duration=longest", output_path])
    subprocess.run(command)
    
# Function to generate and send illustration
async def generate_and_send_illustration(ctx):
    if last_gpt_response:
        dall_e_prompt = f"From the following paragraph, identify and illustrate the most visually striking or significant scene: {last_gpt_response}"
        
        await ctx.send("Generating an illustration, please wait...")
        
        image_url = await generate_image(dall_e_prompt)
        if image_url:
            image_data = await download_image(image_url)
            if image_data:
                await ctx.send(file=File(image_data, 'illustration.png'))
            else:
                await ctx.send("Failed to download the illustration.")
        else:
            await ctx.send("Failed to generate an illustration.")
    else:
        await ctx.send("No recent AI response available to illustrate.")

# Slash command to illustrate the last GPT-4 response
@slash_command(name="illustrate", description="Generate an image based on the last AI response")
async def illustrate(ctx):
    if not last_gpt_response:
        await ctx.send("No recent AI response available to illustrate.")
        return

    # Pretext for DALL-E 3
    dall_e_prompt = f"From the following paragraph, identify and illustrate the most visually striking or significant scene: {last_gpt_response}"

    await ctx.send("Generating an illustration, please wait...")

    image_url = await generate_image(dall_e_prompt)
    if image_url:
        image_data = await download_image(image_url)
        if image_data:
            await ctx.send(file=File(image_data, 'illustration.png'))
        else:
            await ctx.send("Failed to download the illustration.")
    else:
        await ctx.send("Failed to generate an illustration.")

# Function to process text input and generate response and illustration
async def process_text_input(ctx, text_input):
    # Get response from GPT-4 assistant
    gpt_response = await ask_assistant(client, text_input, current_assistant["assistant_id"], current_assistant["thread_id"])
    gpt_text_response = gpt_response.text if hasattr(gpt_response, 'text') else str(gpt_response)

    # Update the last GPT-4 response
    update_last_gpt_response(gpt_text_response)

    # Send GPT-4 response in parts if needed
    gpt_messages = [gpt_text_response[i:i+2000] for i in range(0, len(gpt_text_response), 2000)]
    for msg in gpt_messages:
        await ctx.channel.send(f"**AI Output:** {msg}")

    # Automatically generate and send illustration
    await generate_and_send_illustration(ctx)

# Function to update the last GPT-4 response
def update_last_gpt_response(response):
    global last_gpt_response
    last_gpt_response = response

# Synchronous functions for OpenAI operations
def sync_ask_assistant(client, question, assistant_id, thread_id):

    # Add the user's question to the thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=question
    )

    # Create a Run with the Assistant and the Thread
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )

    # Wait for the Run to complete
    while run.status != 'completed':
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )

    # Retrieve and return the response from the Assistant
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    for message in messages.data:
        if message.role == "assistant":
            # Extract only the text value from the message content
            for content in message.content:
                if content.type == 'text':
                    return content.text.value

def sync_speech_to_text(client, file_path):
    with open(file_path, "rb") as audio_file:
        transcript_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript_response.text

def sync_text_to_speech(client, text, output_file_name):
    if not text:
        raise ValueError("No text provided for text-to-speech conversion.")

    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text
    )
    ai_audio_file_path = os.path.join(ai_output_path, output_file_name)
    with open(ai_audio_file_path, "wb") as file:
        file.write(response.content)
    return ai_audio_file_path
        

async def generate_image(prompt):
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            response = await loop.run_in_executor(pool, lambda: client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1792x1024",
                quality="standard",
                n=1
            ))
            if response.data:
                return response.data[0].url
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

async def download_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                return None
            return io.BytesIO(await resp.read())

async def ask_assistant(client, question, assistant_id, thread_id):
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, lambda: sync_ask_assistant(client, question, assistant_id, thread_id))

async def speech_to_text(client, file_path):
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, sync_speech_to_text, client, file_path)

async def text_to_speech(client, text, output_file_name):
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        ai_audio_file_path = os.path.join(ai_output_path, output_file_name)
        await loop.run_in_executor(pool, sync_text_to_speech, client, text, ai_audio_file_path)
        return ai_audio_file_path
    
# Run the bot
bot.start('')