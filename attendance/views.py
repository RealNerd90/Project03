from __future__ import annotations

import base64
import json
from datetime import datetime, date, time as time_cls
import asyncio

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .models import AttendanceRecord
from .face_system import get_system


def welcome(request: HttpRequest) -> HttpResponse:
    """Landing page based on the welcome UI design."""
    return render(request, "index.html")


def dashboard(request: HttpRequest) -> HttpResponse:
    """Render the main dashboard using the existing static HTML as a template."""
    return render(request, "dashboard.html")


def analytics(request: HttpRequest) -> HttpResponse:
    """Render the analytics page."""
    # Load recent attendance entries to show in the analytics history table.
    # This uses the same model backing additional attendance features.
    attendance_records = AttendanceRecord.objects.all().order_by("-date", "-time")[:50]

    return render(request, "analytics.html", {"attendance_records": attendance_records})


def settings_view(request: HttpRequest) -> HttpResponse:
    """Render the settings page."""
    return render(request, "settings.html")


def scanner(request: HttpRequest) -> HttpResponse:
    """Render the face scanner page."""
    return render(request, "scanner.html")


def signin(request: HttpRequest) -> HttpResponse:
    """Render the sign-in (scanner) page."""
    return render(request, "signin.html")


def registration(request: HttpRequest) -> HttpResponse:
    """Render the registration page."""
    return render(request, "registration.html")


def signin_success(request: HttpRequest) -> HttpResponse:
    """Render a success landing page after a successful face sign-in."""
    name = request.GET.get("name", "Visitor")
    date_str = request.GET.get("date")
    time_str = request.GET.get("time")

    # Provide defaults if values are missing
    now = datetime.now()
    if not date_str:
        date_str = now.strftime("%B %d, %Y")
    if not time_str:
        time_str = now.strftime("%I:%M:%S %p")

    # Simple deterministic ID based on name (not secure; for display only)
    try:
        user_id = f"#{abs(hash(name)) % 900000 + 100000}"
    except Exception:
        user_id = "#000000"

    context = {
        "name": name,
        "date": date_str,
        "time": time_str,
        "user_id": user_id,
    }
    return render(request, "signin_success.html", context)


@require_POST
def signin_scan(request: HttpRequest) -> HttpResponse:
    """Handle sign-in via face recognition.

    - If called as a JSON API (image sent from the browser), runs recognition without
      opening any desktop windows and returns JSON result.
    - If called as a standard form POST (legacy), falls back to the existing
      desktop webcam UI (cv2 window).
    """
    system = get_system()

    content_type = request.META.get("CONTENT_TYPE", "")
    if content_type.startswith("application/json"):
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except Exception as exc:
            return JsonResponse({"success": False, "error": f"Invalid JSON: {exc}"}, status=400)

        image_b64 = payload.get("image")
        if not image_b64:
            return JsonResponse({"success": False, "error": "Missing 'image' field"}, status=400)

        # Support data-url style payloads
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as exc:
            return JsonResponse({"success": False, "error": f"Invalid base64 image: {exc}"}, status=400)

        try:
            result = asyncio.run(system.recognize_image_bytes(image_bytes, mark_attendance=True))
        except Exception as exc:
            return JsonResponse({"success": False, "error": str(exc)}, status=500)

        if result.get("recognized"):
            return JsonResponse({
                "success": True,
                "name": result.get("name"),
                "message": result.get("message", ""),
                "box": result.get("box"),
                "timestamp": result.get("timestamp"),
            })

        return JsonResponse({
            "success": False,
            "message": result.get("message", "Face not recognized."),
            "box": result.get("box"),
        })

    # Fallback: legacy desktop webcam mode (opens a cv2 window on the host).
    try:
        matched_name = asyncio.run(
            system.run_live_mode(camera_index=0, mark_attendance=False, return_name=True)
        )
        if matched_name:
            messages.success(request, f"Signed in as {matched_name} using face recognition.")
        else:
            messages.error(request, "Face not recognized. Please try again or register first.")
    except Exception as exc:
        messages.error(request, f"Sign-in failed: {exc}")

    # After scanning, send the user to the dashboard (or you can redirect to signin again).
    return redirect("dashboard")


@require_POST
def register_face(request: HttpRequest) -> HttpResponse:
    """
    Trigger camera-based registration from the web form.

    This will open the native webcam window on the machine where Django is
    running, capture the student's face, and store reference images in the
    FaceNet database. When it finishes, the same registration page is
    re-rendered with a simple status message.
    """
    full_name = (request.POST.get("fullName") or "").strip()
    if not full_name:
        return render(
            request,
            "registration.html",
            {"error": "Please provide a name before registering."},
            status=400,
        )

    system = get_system()
    # Reuse the existing camera-based registration flow (default camera index 0)
    success = system.register_from_camera(full_name, camera_index=0)

    context = {
        "registration_success": success,
        "registered_name": full_name,
    }
    if not success:
        context["error"] = "Registration failed. Please try again in good lighting."

    return render(request, "registration.html", context)


@csrf_exempt
def create_attendance_record(request: HttpRequest) -> JsonResponse:
    """
    Simple JSON API to create an attendance record.

    This allows frontends (or other services) to create records directly via HTTP.
    The face-recognition script will write to the database using the ORM instead
    of calling this endpoint.
    """
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)

    try:
        payload = request.POST or {}
        name = payload.get("name", "").strip()
        status = payload.get("status", AttendanceRecord.STATUS_PRESENT)
        date_str = payload.get("date")
        time_str = payload.get("time")

        if not name:
            return JsonResponse({"detail": "Missing 'name'."}, status=400)

        if date_str:
            parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            parsed_date = date.today()

        if time_str and time_str != "OUT OF RANGE":
            parsed_time = datetime.strptime(time_str, "%H:%M:%S").time()
        else:
            parsed_time = None

        record = AttendanceRecord.objects.create(
            name=name,
            date=parsed_date,
            time=parsed_time,
            status=status,
        )

        return JsonResponse(
            {
                "id": record.id,
                "name": record.name,
                "date": record.date.isoformat(),
                "time": record.time.isoformat() if record.time else None,
                "status": record.status,
            },
            status=201,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return JsonResponse({"detail": f"Error creating record: {exc}"}, status=500)

