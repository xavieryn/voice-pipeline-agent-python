import logging
import json
import asyncio
import websockets
from typing import Dict, Any, Set

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, openai, deepgram, silero, turn_detector

from livekit import rtc
from livekit.agents.llm import ChatMessage, ChatImage



load_dotenv(dotenv_path=".env.local")  # gets all of the api keys
logger = logging.getLogger("voice-agent")

# Store UI websocket connections
ui_connections: Set[websockets.WebSocketServerProtocol] = set()

# UI WebSocket handler
async def ui_websocket_handler(websocket, path):
    ui_connections.add(websocket)
    try:
        async for message in websocket:
            # Handle any messages from the UI if needed
            logger.info(f"Received message from UI: {message}")
    finally:
        ui_connections.remove(websocket)

# Start the WebSocket server for UI updates
async def start_ui_websocket_server():
    server = await websockets.serve(ui_websocket_handler, "localhost", 8765)
    logger.info("UI WebSocket server started on port 8765")
    return server

# Function to send UI updates
async def send_ui_update(update_type: str, data: Dict[str, Any]):
    if not ui_connections:
        logger.warning("No UI connections available to send update")
        return
    
    message = json.dumps({"type": update_type, "data": data})
    websockets.broadcast(ui_connections, message)
    logger.info(f"Sent UI update: {message}")

# Define function specifications for the LLM
function_specs = [
    {
        "name": "change_background_color",
        "description": "Change the background color of the interface",
        "parameters": {
            "type": "object",
            "properties": {
                "color": {
                    "type": "string",
                    "description": "The color to change the background to (e.g., 'red', 'blue', 'green', '#FF0000')"
                }
            },
            "required": ["color"]
        }
    },
    {
        "name": "change_text_size",
        "description": "Change the size of text in the interface",
        "parameters": {
            "type": "object",
            "properties": {
                "size": {
                    "type": "string",
                    "description": "The size to change the text to (e.g., 'small', 'medium', 'large')"
                }
            },
            "required": ["size"]
        }
    }
]

async def get_video_track(room: rtc.Room):
    """Find and return the first available remote video track in the room."""
    for participant_id, participant in room.remote_participants.items():
        for track_id, track_publication in participant.track_publications.items():
            if track_publication.track and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                logger.info(
                    f"Found video track {track_publication.track.sid} "
                    f"from participant {participant_id}"
                )
                return track_publication.track
    raise ValueError("No remote video track found in the room")

async def get_latest_image(room: rtc.Room):
    """Capture and return a single frame from the video track."""
    video_stream = None
    try:
        video_track = await get_video_track(room)
        video_stream = rtc.VideoStream(video_track)
        async for event in video_stream:
            logger.debug("Captured latest video frame")
            return event.frame
    except Exception as e:
        logger.error(f"Failed to get latest image: {e}")
        return None
    finally:
        if video_stream:
            await video_stream.aclose()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    # Store the WebSocket server in proc.userdata
    proc.userdata["websocket_task"] = None

async def entrypoint(ctx: JobContext):
    async def before_llm_cb(assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext):
        """
        Callback that runs right before the LLM generates a response.
        Captures the current video frame and adds it to the conversation context.
        """
        latest_image = await get_latest_image(ctx.room)
        if latest_image:
            image_content = [ChatImage(image=latest_image)]
            chat_ctx.messages.append(ChatMessage(role="user", content=image_content))
            logger.debug("Added latest frame to conversation context")

    # Start the WebSocket server for UI updates
    websocket_server_task = asyncio.create_task(start_ui_websocket_server())
    ctx.proc.userdata["websocket_task"] = websocket_server_task
    
    initial_ctx = llm.ChatContext().append(
    role="system",
    text=(
        "You are a voice assistant created by LiveKit that can both see and hear. "
        "You should use short and concise responses, avoiding unpronounceable punctuation. "
        "When you see an image in our conversation, naturally incorporate what you see "
        "into your response. Keep visual descriptions brief but informative."
    ),
)

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    # Wait for the first participant to connect

    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # Initialize the agent with function calling capabilities
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
        before_llm_cb=before_llm_cb
)
    # agent = VoicePipelineAgent( # this is surprisingly fast considering the fact that it doesn't use speech to speech
    #     vad=ctx.proc.userdata["vad"],
    #     stt=deepgram.STT(),
    #     llm=openai.LLM(model="gpt-4o-mini"), # TEST with different models
    #     tts=cartesia.TTS(),
    #     turn_detector=turn_detector.EOUModel(), # Will detect when we interrupt the person
    #     # minimum delay for endpointing, used when turn detector believes the user is done with their turn
    #     min_endpointing_delay=0.5,
    #     # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn
    #     max_endpointing_delay=5.0,
    #     chat_ctx=initial_ctx,
    # )
    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    # # Handle function calls from the LLM
    # @agent.on("function_call")
    # async def on_function_call(function_name: str, arguments: Dict[str, Any]):
    #     logger.info(f"Function call: {function_name} with arguments: {arguments}")
        
    #     if function_name == "change_background_color":
    #         color = arguments.get("color")
    #         await send_ui_update("change_background", {"color": color})
    #         await agent.say(f"I've changed the background color to {color}.", allow_interruptions=True)
        
    #     elif function_name == "change_text_size":
    #         size = arguments.get("size")
    #         await send_ui_update("change_text_size", {"size": size})
    #         await agent.say(f"I've changed the text size to {size}.", allow_interruptions=True)
        
    #     else:
    #         await agent.say("I don't know how to do that yet.", allow_interruptions=True)

    agent.start(ctx.room, participant)

    # # The agent should be polite and greet the user when it joins
    # await agent.say("Hey, how can I help you today? I can change the background color or text size if you'd like.", allow_interruptions=True)
    
    try:
        # Keep the agent running until the context is done
        await ctx.done()
    finally:
        # Clean up the WebSocket server when done
        if websocket_server_task and not websocket_server_task.done():
            websocket_server_task.cancel()
            try:
                await websocket_server_task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )