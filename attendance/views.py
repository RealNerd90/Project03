from __future__ import annotations

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
    return render(request, "analytics.html")


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


@require_POST
def signin_scan(request: HttpRequest) -> HttpResponse:
    """
    Use existing face data to sign a user in by running the live
    attendance scan (webcam). This reuses run_live_mode, which will
    log attendance into the Django database via log_attendance.
    """
    system = get_system()

    try:
        # Run live mode but do NOT mark attendance; just get the matched name.
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

