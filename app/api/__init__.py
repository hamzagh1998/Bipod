import asyncio
import contextlib
import json
from fastapi import APIRouter, HTTPException, Body, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from typing import Any, Dict, List, Optional
from app.services.brain_service import brain_service
from app.services.memory_service import memory_service
from app.services.auth_service import auth_service
from app.services.studio_service import studio_service
from app.core.config import settings
from app.api.schemas import * # We'll use specific imports below to be safe

router = APIRouter()


def _encode_stream_event(event_type: str, **payload: Any) -> bytes:
    return (json.dumps({"type": event_type, **payload}, ensure_ascii=True) + "\n").encode("utf-8")

# --- Auth Models ---
from app.api.schemas import (
    UserAuth, Token, UserResponse, ConversationResponse, MessageResponse,
    ChatRequest, ChatResponse, ConversationUpdate, ArchiveUnlock,
    StudioProjectCreate, StudioProjectResponse, StudioImageResponse,
    StudioPromptImproveRequest, StudioPromptImproveResponse,
)

@router.get("/health")
async def health_check():
    return {"status": "ok", "message": "Bipod's nervous system is functional."}

@router.get("/system/config")
async def get_system_config():
    """Returns hardware capabilities and available models."""
    try:
        # Proxy detailed hardware info from Imagine service
        async with httpx.AsyncClient(timeout=3.0) as client:
            # Parallel fetch for speed
            r_sys, r_models = await asyncio.gather(
                client.get(f"{settings.IMAGINE_API_URL}/system"),
                client.get(f"{settings.IMAGINE_API_URL}/models")
            )
            
            imagine_system = r_sys.json() if r_sys.status_code == 200 else {}
            imagine_models = r_models.json().get("models", []) if r_models.status_code == 200 else []
            vram_tier = imagine_system.get("vram_tier")
            total_vram = imagine_system.get("vram", {}).get("total_gb", 0)

    except Exception:
        # Fallback if Imagine service is down
        imagine_system = {}
        imagine_models = []
        vram_tier = None
        total_vram = settings.GPU_VRAM

    return {
        "hardware": settings.HARDWARE_TARGET,
        "use_gpu": settings.USE_GPU,
        "gpu_name": imagine_system.get("gpu_name") or settings.GPU_NAME,
        "gpu_vram": round(total_vram, 2),
        "vram_tier": vram_tier,
        "active_brain_model": settings.ACTIVE_MODEL,
        "available_brain_models": [
            {"id": settings.SMART_MODEL, "name": "Smart (7b)", "tier": "Heavy", "req": "8GB+ VRAM", "available": True},
            {"id": settings.HEAVY_MODEL, "name": "Heavy (8b)", "tier": "Heavy", "req": "8GB+ VRAM/RAM", "available": True},
            {"id": settings.MEDIUM_MODEL, "name": "Medium (3b)", "tier": "Medium", "req": "4GB+ RAM", "available": True},
            {
                "id": settings.LIGHT_MODEL,
                "name": "Light (1b, edge only)",
                "tier": "Light",
                "req": "1GB+ RAM",
                "available": settings.HARDWARE_TARGET == "arm64",
            },
        ],
        "active_imagine_model": settings.ACTIVE_IMAGINE_MODEL,
        "available_imagine_models": imagine_models if imagine_models else [
             # Fallback list if service is unreachable
            {"id": "sdxl-lightning", "name": "SDXL Lightning (Fast)", "req": "8GB+ VRAM", "supports_img2img": True, "supports_negative_prompt": True},
            {"id": "stable-diffusion", "name": "Realistic Vision V6", "req": "4GB+ VRAM", "supports_img2img": True, "supports_negative_prompt": True},
            {"id": "flux-schnell", "name": "Flux.1-schnell (4-bit, Photorealism)", "req": "10GB+ VRAM", "supports_img2img": False, "supports_negative_prompt": False, "available": total_vram >= 5.5},
            {"id": "dalle-mini", "name": "Tiny-SD (CPU)", "req": "None (CPU)", "supports_img2img": True, "supports_negative_prompt": True}
        ]
    }

# --- Auth Endpoints ---
@router.post("/auth/signup", response_model=Token)
async def signup(user_data: UserAuth):
    existing_user = await memory_service.get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = auth_service.get_password_hash(user_data.password)
    user = await memory_service.create_user(user_data.username, hashed_password)
    
    access_token = auth_service.create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token}

@router.post("/auth/login", response_model=Token)
async def login(user_data: UserAuth):
    user = await memory_service.get_user_by_username(user_data.username)
    if not user or not auth_service.verify_password(user_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = auth_service.create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token}

@router.get("/auth/me")
async def get_me(current_user_id: int = Depends(auth_service.get_current_user)):
    user = await memory_service.get_user_by_id(current_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user.id, "username": user.username}

# --- Protected Chat Endpoints ---
@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(user_id: int = Depends(auth_service.get_current_user)):
    return await memory_service.get_conversations(user_id)

@router.post("/conversations")
async def create_conversation(
    title: str = Body(..., embed=True),
    user_id: int = Depends(auth_service.get_current_user)
):
    conv_id = await memory_service.create_conversation(user_id, title)
    return {"id": conv_id, "title": title}

@router.get("/conversations/{conv_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    conv_id: str,
    user_id: int = Depends(auth_service.get_current_user)
):
    return await memory_service.get_messages(conv_id, user_id)

@router.patch("/conversations/{conv_id}")
async def update_conversation(
    conv_id: str, 
    update: ConversationUpdate,
    user_id: int = Depends(auth_service.get_current_user)
):
    await memory_service.update_conversation(
        conv_id, 
        user_id,
        title=update.title, 
        is_archived=update.is_archived, 
        password=update.password
    )
    return {"status": "success"}

@router.post("/conversations/{conv_id}/unlock")
async def unlock_conversation(
    conv_id: str, 
    unlock: ArchiveUnlock,
    user_id: int = Depends(auth_service.get_current_user)
):
    is_valid = await memory_service.verify_archive_password(conv_id, user_id, unlock.password)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid password")
    return {"status": "success"}

@router.delete("/conversations/{conv_id}")
async def delete_conversation(
    conv_id: str,
    user_id: int = Depends(auth_service.get_current_user)
):
    await memory_service.delete_conversation(conv_id, user_id)
    return {"status": "success"}

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: int = Depends(auth_service.get_current_user)
):
    """The main entry point for talking to Bipod."""
    try:
        # Check if conversation exists and belongs to user
        conv = await memory_service.get_conversation(request.conversation_id, user_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        response_text = await brain_service.think(
            request.message, 
            request.conversation_id,
            user_id,
            model_id=request.model_id,
            reasoning_mode=request.reasoning_mode,
            imagine_model=request.imagine_model,
            attachments=[a.model_dump() for a in request.attachments] if request.attachments else None
        )
        return ChatResponse(response=response_text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    user_id: int = Depends(auth_service.get_current_user),
):
    conv = await memory_service.get_conversation(payload.conversation_id, user_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    async def emit_progress(event: str, data: Dict[str, Any]) -> None:
        await queue.put({"type": event, **data})

    async def run_chat() -> None:
        try:
            response_text = await brain_service.think(
                payload.message,
                payload.conversation_id,
                user_id,
                model_id=payload.model_id,
                reasoning_mode=payload.reasoning_mode,
                imagine_model=payload.imagine_model,
                attachments=[a.model_dump() for a in payload.attachments] if payload.attachments else None,
                progress_callback=emit_progress,
            )
            await queue.put({"type": "response", "text": response_text})
        except HTTPException as exc:
            await queue.put({"type": "error", "detail": exc.detail, "status_code": exc.status_code})
        except Exception as exc:
            await queue.put({"type": "error", "detail": str(exc), "status_code": 500})
        finally:
            await queue.put({"type": "done"})

    async def stream_events():
        task = asyncio.create_task(run_chat())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                event_type = event.pop("type", "status")
                yield _encode_stream_event(event_type, **event)
                if event_type in {"done", "error"}:
                    break
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    return StreamingResponse(
        stream_events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/clear")
async def clear_memory(
    conversation_id: str = Body(..., embed=True),
    user_id: int = Depends(auth_service.get_current_user)
):
    # Verify owner before clearing
    conv = await memory_service.get_conversation(conversation_id, user_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await brain_service.clear_memory(conversation_id)
    return {"status": "success", "message": f"Bipod has forgotten the recent past in {conversation_id}."}


@router.post("/generate")
async def generate_image(
    payload: dict = Body(...),
    user_id: int = Depends(auth_service.get_current_user)
):
    """Proxy image generation to the Imagine service for Studio UI."""
    try:
        project_id = payload.pop("project_id", None)
        if project_id and not await studio_service.get_project(project_id, user_id):
            raise HTTPException(status_code=404, detail="Project not found")
        await brain_service._unload_ollama()
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(f"{settings.IMAGINE_API_URL}/generate", json=payload)

        try:
            content = resp.json()
        except ValueError:
            content = {"detail": resp.text}

        if (
            resp.status_code == 200
            and project_id
            and content.get("status") == "success"
            and content.get("image_base64")
        ):
            saved_image = await studio_service.add_project_image(
                project_id=project_id,
                user_id=user_id,
                image_base64=content["image_base64"],
                mime_type=content.get("mime_type", "image/jpeg"),
                file_extension=content.get("file_extension", "jpg"),
                metadata=content,
            )
            content["saved_image"] = saved_image

        return JSONResponse(status_code=resp.status_code, content=jsonable_encoder(content))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Imagine service unavailable: {str(e)}")


@router.post("/upscale")
async def upscale_image(
    payload: dict = Body(...),
    user_id: int = Depends(auth_service.get_current_user)
):
    """Proxy AI upscaling to the Imagine service for Studio UI."""
    try:
        project_id = payload.pop("project_id", None)
        if project_id and not await studio_service.get_project(project_id, user_id):
            raise HTTPException(status_code=404, detail="Project not found")
        await brain_service._unload_ollama()
        async with httpx.AsyncClient(timeout=1200.0) as client:
            resp = await client.post(f"{settings.IMAGINE_API_URL}/upscale", json=payload)

        try:
            content = resp.json()
        except ValueError:
            content = {"detail": resp.text}

        if (
            resp.status_code == 200
            and project_id
            and content.get("status") == "success"
            and content.get("image_base64")
        ):
            saved_image = await studio_service.add_project_image(
                project_id=project_id,
                user_id=user_id,
                image_base64=content["image_base64"],
                mime_type=content.get("mime_type", "image/jpeg"),
                file_extension=content.get("file_extension", "jpg"),
                metadata=content,
            )
            content["saved_image"] = saved_image

        return JSONResponse(status_code=resp.status_code, content=jsonable_encoder(content))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Imagine service unavailable: {str(e)}")


@router.get("/studio/projects", response_model=List[StudioProjectResponse])
async def list_studio_projects(user_id: int = Depends(auth_service.get_current_user)):
    return await studio_service.list_projects(user_id)


@router.post("/studio/projects", response_model=StudioProjectResponse)
async def create_studio_project(
    payload: StudioProjectCreate,
    user_id: int = Depends(auth_service.get_current_user),
):
    project = await studio_service.create_project(user_id, payload.title)
    return {
        "id": project.id,
        "title": project.title,
        "created_at": project.created_at,
        "image_count": 0,
        "cover_image_url": None,
    }


@router.delete("/studio/projects/{project_id}")
async def delete_studio_project(
    project_id: str,
    user_id: int = Depends(auth_service.get_current_user),
):
    deleted = await studio_service.delete_project(project_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "success"}


@router.get("/studio/projects/{project_id}/images", response_model=List[StudioImageResponse])
async def list_studio_project_images(
    project_id: str,
    user_id: int = Depends(auth_service.get_current_user),
):
    project = await studio_service.get_project(project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return await studio_service.list_project_images(project_id, user_id)


@router.delete("/studio/projects/{project_id}/images/{image_id}")
async def delete_studio_project_image(
    project_id: str,
    image_id: str,
    user_id: int = Depends(auth_service.get_current_user),
):
    deleted = await studio_service.delete_project_image(project_id, image_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"status": "success"}


@router.post("/studio/prompt-improve", response_model=StudioPromptImproveResponse)
async def improve_studio_prompt(
    payload: StudioPromptImproveRequest,
    user_id: int = Depends(auth_service.get_current_user),
):
    _ = user_id
    try:
        return await brain_service.improve_studio_prompts(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt or "",
            model_type=payload.model_type,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Prompt improvement failed: {str(e)}")
