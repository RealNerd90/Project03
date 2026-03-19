from __future__ import annotations

import base64
import json
import os
from datetime import datetime, date, time as time_cls, timedelta
import asyncio
import calendar

from io import BytesIO
import uuid
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from django.utils import timezone
from django.utils.text import slugify
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .models import AttendanceRecord, RegisteredUser, AdminAccount
from .face_system import get_system
from attendance_system import check_location_allowed


NAME_ALIASES: dict[str, str] = {
    # Fix common registration/label typos from face DB folder names.
    # Key matching is case-insensitive.
    "amatya": "Amartya",
}


def _normalize_display_name(name: str | None) -> str:
    raw = (name or "").strip()
    if not raw:
        return "Visitor"
    mapped = NAME_ALIASES.get(raw.lower())
    return mapped or raw


def _profile_photo_url_for(display_name: str | None) -> str:
    safe_name = _normalize_display_name(display_name)
    if safe_name == "Visitor":
        return ""
    profile_photo_folder = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "frontend",
        "profile_photos",
    )
    photo_filename = f"{slugify(safe_name)}.jpg"
    profile_photo_path = os.path.join(profile_photo_folder, photo_filename)
    return f"/static/profile_photos/{photo_filename}" if os.path.exists(profile_photo_path) else ""


def welcome(request: HttpRequest) -> HttpResponse:
    """Landing page based on the welcome UI design."""
    return render(request, "index.html")


def dashboard(request: HttpRequest) -> HttpResponse:
    """Render the main dashboard using the existing static HTML as a template."""
    today = timezone.localdate()
    display_name = _normalize_display_name(request.session.get("display_name"))
    profile_photo_url = _profile_photo_url_for(display_name)

    # Build a "recent activity" feed from real DB records.
    # We scope everything to the current signed-in name stored in the session,
    # so each person only sees their own history.
    if display_name == "Visitor":
        records = AttendanceRecord.objects.none()
    else:
        records = (
            AttendanceRecord.objects.filter(name=display_name)
            .order_by("-date", "-created_at")[:25]
        )

    def _format_time(t: time_cls | None) -> str:
        if not t:
            return "--"
        return t.strftime("%I:%M %p").lstrip("0")

    def _format_day(d: date) -> str:
        # e.g. "Friday, Oct 20"
        return d.strftime("%A, %b %d").replace(" 0", " ")

    def _format_duration(start: time_cls | None, end: time_cls | None) -> str | None:
        if not start or not end:
            return None
        start_dt = datetime.combine(date.today(), start)
        end_dt = datetime.combine(date.today(), end)
        if end_dt < start_dt:
            # Crossed midnight
            end_dt = end_dt.replace(day=end_dt.day + 1)
        delta = end_dt - start_dt
        total_minutes = int(delta.total_seconds() // 60)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours}h {minutes}m"

    events: list[dict[str, str]] = []
    for r in records:
        record_name = _normalize_display_name(r.name)
        # Newest-first: check-out is after check-in, so add it first.
        if r.check_out_time:
            duration = _format_duration(r.time, r.check_out_time)
            events.append(
                {
                    "kind": "checkout",
                    "title": "Checked Out",
                    "date_label": _format_day(r.date),
                    "time_label": _format_time(r.check_out_time),
                    "meta": f"{record_name} • Duration: {duration}" if duration else f"{record_name} • {r.get_status_display()}",
                }
            )
        if r.time:
            events.append(
                {
                    "kind": "checkin",
                    "title": "Checked In",
                    "date_label": _format_day(r.date),
                    "time_label": _format_time(r.time),
                    "meta": f"{record_name} • {r.get_status_display()}",
                }
            )

        if len(events) >= 3:
            break

    # Weekly totals (Mon-Sun) for this user only
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)
    if display_name == "Visitor":
        week_records = AttendanceRecord.objects.none()
    else:
        week_records = AttendanceRecord.objects.filter(
            name=display_name, date__gte=week_start, date__lte=week_end
        )

    total_seconds = 0
    for r in week_records:
        if r.status != AttendanceRecord.STATUS_PRESENT:
            continue
        if not r.time or not r.check_out_time:
            continue
        start_dt = datetime.combine(r.date, r.time)
        end_dt = datetime.combine(r.date, r.check_out_time)
        if end_dt < start_dt:
            end_dt += timedelta(days=1)
        total_seconds += int((end_dt - start_dt).total_seconds())

    total_hours = total_seconds / 3600.0
    weekly_goal_hours = 40.0
    progress_pct = 0 if weekly_goal_hours <= 0 else int(round(min(100.0, (total_hours / weekly_goal_hours) * 100.0)))
    remaining_hours = max(0.0, weekly_goal_hours - total_hours)

    weekly_hours_text = f"{total_hours:.1f} Hrs"
    remaining_text = f"{remaining_hours:.1f} hours left to reach your weekly goal."

    # Calendar (current month)
    cal = calendar.Calendar(firstweekday=6)  # Sunday first to match UI (SU..SA)
    year = today.year
    month = today.month
    calendar_month_label = today.strftime("%B %Y")
    month_weeks = cal.monthdatescalendar(year, month)
    calendar_cells: list[dict[str, str]] = []

    if display_name == "Visitor":
        present_qs = AttendanceRecord.objects.none()
    else:
        present_qs = AttendanceRecord.objects.filter(
            name=display_name,
            date__year=year,
            date__month=month,
            status=AttendanceRecord.STATUS_PRESENT,
        )
    present_days = set(present_qs.values_list("date", flat=True))

    for week in month_weeks:
        for day in week:
            in_month = day.month == month
            is_today = day == today
            is_present = day in present_days

            classes = ["calendar-cell"]
            if not in_month:
                classes.append("text-muted")
            else:
                if is_today:
                    classes.append("bg-primary")
                    classes.append("text-white")
                    classes.append("rounded-md")
                    classes.append("font-bold")
                    classes.append("shadow-sm")
                elif is_present:
                    classes.append("bg-green-light")
                    classes.append("rounded-md")
                    classes.append("font-medium")

            calendar_cells.append(
                {
                    "label": str(day.day) if in_month else "",
                    "classes": " ".join(classes),
                }
            )

    return render(
        request,
        "dashboard.html",
        {
            "display_name": display_name,
            "profile_photo_url": profile_photo_url,
            "recent_activity": events[:3],
            "weekly_hours_text": weekly_hours_text,
            "weekly_progress_pct": progress_pct,
            "weekly_remaining_text": remaining_text,
            "calendar_cells": calendar_cells,
            "calendar_month_label": calendar_month_label,
        },
    )


def admin_dashboard(request: HttpRequest) -> HttpResponse:
    """Admin dashboard UI (static template in frontend/)."""
    if not request.session.get("is_admin"):
        return redirect("manual-login")
    return render(request, "admin_dashboard.html")


def analytics(request: HttpRequest) -> HttpResponse:
    """Render the analytics page."""
    # Load recent attendance entries for the current signed-in person only.
    display_name = _normalize_display_name(request.session.get("display_name"))
    profile_photo_url = _profile_photo_url_for(display_name)
    if display_name == "Visitor":
        attendance_records = AttendanceRecord.objects.none()
    else:
        attendance_records = (
            AttendanceRecord.objects.filter(name=display_name)
            .order_by("-date", "-time")[:50]
        )

    return render(
        request,
        "analytics.html",
        {
            "attendance_records": attendance_records,
            "profile_photo_url": profile_photo_url,
            "display_name": display_name,
        },
    )


def logout_view(request: HttpRequest) -> HttpResponse:
    """POST-only logout; UI confirmation is handled client-side via a modal."""
    if request.method != "POST":
        return redirect("dashboard")

    try:
        request.session.flush()
    except Exception:
        request.session.clear()
    return redirect("signin")


def profile_view(request: HttpRequest) -> HttpResponse:
    """Render the profile page based on the provided UI design."""
    display_name = _normalize_display_name(request.session.get("display_name"))
    profile_photo_url = _profile_photo_url_for(display_name)

    user = None
    email = ""
    phone = ""
    dob_display = ""
    gender = ""
    role_key = request.session.get("account_role", "")
    if display_name != "Visitor":
        user, _ = RegisteredUser.objects.get_or_create(name=display_name)
        email = user.email or ""
        phone = user.phone or ""
        dob_display = user.dob_display or ""
        gender = user.gender or ""
        if user.account_role:
            role_key = user.account_role

    # Deterministic employee id for display only.
    try:
        emp_id = f"EMP-{abs(hash(display_name)) % 9000 + 1000}"
    except Exception:
        emp_id = "EMP-1234"

    # Account role is stored in the DB/session. If missing, keep blank (user fills it in).
    role_map = {
        "student": ("Student", "Access to courses"),
        "employee": ("Employee", "Standard portal access"),
        "teacher": ("Teacher", "Management tools"),
    }
    role_label, role_desc = role_map.get(role_key, ("—", ""))

    def _pretty_dob(val: str) -> str:
        raw = (val or "").strip()
        if not raw:
            return ""
        # Accept yyyy-mm-dd, mm/dd/yyyy, or "Month DD, YYYY"
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y"):
            try:
                d = datetime.strptime(raw, fmt).date()
                return d.strftime("%B %d, %Y").replace(" 0", " ")
            except Exception:
                continue
        return raw

    # Keep fields blank until user fills them in.
    dob_display = _pretty_dob(dob_display) or _pretty_dob(request.session.get("profile_dob_display", ""))
    gender = gender or ""
    phone = phone or ""

    # The design is mostly static; we provide sensible defaults.
    return render(
        request,
        "profile.html",
        {
            "display_name": display_name,
            "profile_photo_url": profile_photo_url,
            "profile_name": display_name,
            "emp_id": emp_id,
            "role_title": "Senior Product Designer",
            "full_name": display_name,
            "date_of_birth": dob_display,
            "gender": gender,
            "email": email,
            "phone": phone,
            "account_role_label": role_label,
            "account_role_desc": role_desc,
        },
    )


def profile_edit_view(request: HttpRequest) -> HttpResponse:
    """Render the edit profile form page."""
    display_name = _normalize_display_name(request.session.get("display_name"))
    profile_photo_url = _profile_photo_url_for(display_name)

    user = None
    email = ""
    phone = ""
    dob_display = ""
    gender = ""
    if display_name != "Visitor":
        user, _ = RegisteredUser.objects.get_or_create(name=display_name)
        email = user.email or ""
        phone = user.phone or ""
        dob_display = user.dob_display or ""
        gender = user.gender or ""

    def _dob_to_iso(val: str) -> str:
        raw = (val or "").strip()
        if not raw:
            return ""
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y"):
            try:
                d = datetime.strptime(raw, fmt).date()
                return d.isoformat()
            except Exception:
                continue
        return ""

    if request.method == "POST":
        photo_action = (request.POST.get("photo_action") or "").strip()
        if photo_action in {"upload_profile_photo", "remove_profile_photo"} and display_name != "Visitor":
            profile_photo_folder = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "frontend",
                "profile_photos",
            )
            os.makedirs(profile_photo_folder, exist_ok=True)
            photo_filename = f"{slugify(display_name)}.jpg"
            profile_photo_path = os.path.join(profile_photo_folder, photo_filename)

            if photo_action == "upload_profile_photo":
                upload = request.FILES.get("profile_photo")
                if upload:
                    try:
                        with open(profile_photo_path, "wb") as f:
                            for chunk in upload.chunks():
                                f.write(chunk)
                    except Exception:
                        pass
                return redirect("profile-edit")

            if photo_action == "remove_profile_photo":
                try:
                    if os.path.exists(profile_photo_path):
                        os.remove(profile_photo_path)
                except Exception:
                    pass
                return redirect("profile-edit")

        full_name = (request.POST.get("full_name") or display_name).strip()
        email_val = (request.POST.get("email") or "").strip()
        dob_val = (request.POST.get("dob") or "").strip()
        dob_iso = _dob_to_iso(dob_val) or dob_val
        gender_val = (request.POST.get("gender") or "").strip()
        role_key = (request.POST.get("role") or "employee").strip().lower()
        if role_key not in {"student", "employee", "teacher"}:
            role_key = "employee"

        # Persist name/email into the registry if possible.
        if user and full_name and full_name != "Visitor":
            # Name/email are fixed; we do not mutate them here.
            if dob_iso:
                user.dob_display = dob_iso
            if gender_val:
                user.gender = gender_val
            if phone:
                user.phone = phone
            if email_val:
                # Only fill email if it's empty today.
                if not user.email:
                    user.email = email_val
            user.account_role = role_key
            user.save()

        # Store account role preference and DOB in the session.
        request.session["account_role"] = role_key
        if dob_iso:
            request.session["profile_dob_display"] = dob_iso

        return redirect("profile")

    current_role = (user.account_role if user and user.account_role else None) or request.session.get("account_role", "employee")
    dob_iso = _dob_to_iso(dob_display) or _dob_to_iso(request.session.get("profile_dob_display", "")) or "1992-05-15"
    context = {
        "display_name": display_name,
        "profile_photo_url": profile_photo_url,
        "full_name": display_name,
        "email": email or "alex.henderson@company.com",
        "phone": phone or "+1 (555) 000-1234",
        "date_of_birth": dob_display or "05/15/1992",
        "dob_iso": dob_iso,
        "gender": gender or "Male",
        "current_role": current_role,
    }

    return render(request, "profile_edit.html", context)


def settings_view(request: HttpRequest) -> HttpResponse:
    """Render and update the settings page with notification sound and profile photo handlers."""
    display_name = _normalize_display_name(request.session.get("display_name"))
    email = ""
    if display_name != "Visitor":
        try:
            user = RegisteredUser.objects.filter(name=display_name).first()
            if user and user.email:
                email = user.email
        except RegisteredUser.DoesNotExist:
            email = ""

    # Handle profile photo upload/remove actions
    profile_photo_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "profile_photos")
    os.makedirs(profile_photo_folder, exist_ok=True)
    photo_filename = f"{slugify(display_name)}.jpg"
    profile_photo_path = os.path.join(profile_photo_folder, photo_filename)
    profile_photo_url = f"/static/profile_photos/{photo_filename}" if os.path.exists(profile_photo_path) else ""

    if request.method == "POST":
        action = request.POST.get("action")
        if action == "upload_profile_photo" and display_name != "Visitor":
            upload = request.FILES.get("profile_photo")
            if upload:
                try:
                    with open(profile_photo_path, "wb") as f:
                        for chunk in upload.chunks():
                            f.write(chunk)
                    profile_photo_url = f"/static/profile_photos/{photo_filename}"
                    messages.success(request, "Profile photo updated.")
                except Exception:
                    messages.error(request, "Could not upload profile photo.")
            else:
                messages.error(request, "Please choose an image to upload.")
            return redirect("settings")

        if action == "remove_profile_photo" and display_name != "Visitor":
            if os.path.exists(profile_photo_path):
                os.remove(profile_photo_path)
                messages.success(request, "Profile photo removed.")
            else:
                messages.info(request, "No profile photo to remove.")
            return redirect("settings")

    sound_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notification sounds")
    notification_sounds = []
    if os.path.isdir(sound_dir):
        files = [name for name in os.listdir(sound_dir) if name.lower().endswith((".wav", ".mp3", ".ogg"))]

        def _natural_key(name: str):
            import re
            m = re.search(r"(\d+)", name)
            if m:
                return (name[:m.start()].lower(), int(m.group(1)), name)
            return (name.lower(), 0, name)

        notification_sounds = sorted(files, key=_natural_key)

    selected_sound = request.session.get("notification_sound") or (notification_sounds[0] if notification_sounds else "")
    if selected_sound not in notification_sounds and notification_sounds:
        selected_sound = notification_sounds[0]

    return render(
        request,
        "settings.html",
        {
            "display_name": display_name,
            "email": email,
            "notification_sounds": notification_sounds,
            "selected_sound": selected_sound,
            "profile_photo_url": profile_photo_url,
        },
    )


def scanner(request: HttpRequest) -> HttpResponse:
    """Render the face scanner page."""
    display_name = _normalize_display_name(request.session.get("display_name"))
    return render(
        request,
        "scanner.html",
        {"profile_photo_url": _profile_photo_url_for(display_name)},
    )


def signin(request: HttpRequest) -> HttpResponse:
    """Render the sign-in (scanner) page."""
    return render(request, "signin.html")


def manual_login(request: HttpRequest) -> HttpResponse:
    """Manual email/password login (simple registry-based)."""
    if request.method == "POST":
        email = (request.POST.get("email") or "").strip()
        password = (request.POST.get("password") or "").strip()
        if not email or not password:
            return render(
                request,
                "manual_login.html",
                {"error": "Please enter both email and password.", "email": email},
                status=400,
            )

        admin = AdminAccount.objects.filter(email__iexact=email).first()
        if admin and admin.check_password(password):
            request.session["is_admin"] = True
            request.session["display_name"] = "Admin"
            return redirect("admin-dashboard")

        user = RegisteredUser.objects.filter(email__iexact=email).first()
        if not user:
            return render(
                request,
                "manual_login.html",
                {"error": "No account found for that email.", "email": email},
                status=400,
            )

        # Verify password (stored in attendance_registereduser.password).
        if not user.password or user.password != password:
            return render(
                request,
                "manual_login.html",
                {"error": "Invalid email or password.", "email": email},
                status=400,
            )

        request.session["display_name"] = _normalize_display_name(user.name)
        request.session.pop("is_admin", None)
        return redirect("dashboard")

    return render(request, "manual_login.html")


def registration(request: HttpRequest) -> HttpResponse:
    """Render the registration page."""
    return render(request, "registration.html")


def signin_success(request: HttpRequest) -> HttpResponse:
    """Render a success landing page after a successful face sign-in."""
    name = _normalize_display_name(request.GET.get("name", "Visitor"))
    date_str = request.GET.get("date")
    time_str = request.GET.get("time")
    lat = request.GET.get("lat")
    lon = request.GET.get("lon")
    acc = request.GET.get("acc")

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

    # Persist name for dashboard greeting (no full auth yet).
    request.session["display_name"] = name

    # Account role from database (fallback: employee).
    role_key = ""
    if name != "Visitor":
        user, _ = RegisteredUser.objects.get_or_create(name=name)
        if user.account_role:
            role_key = user.account_role
    role_map = {
        "student": ("Student", "Access to courses"),
        "employee": ("Employee", "Standard portal access"),
        "teacher": ("Teacher", "Management tools"),
    }
    account_role_label, account_role_desc = role_map.get(role_key, ("—", ""))

    context = {
        "name": name,
        "date": date_str,
        "time": time_str,
        "user_id": user_id,
        "lat": lat,
        "lon": lon,
        "acc": acc,
        "account_role_label": account_role_label,
        "account_role_desc": account_role_desc,
    }
    return render(request, "signin_success.html", context)


def attendance_success(request: HttpRequest) -> HttpResponse:
    """
    Render a success page after a check-in / check-out scan.
    Expects query params: name, action ('checkin'|'checkout'), date, time.
    """
    name = _normalize_display_name(request.GET.get("name", "Visitor"))
    action = (request.GET.get("action") or "").strip().lower()
    if action not in {"checkin", "checkout"}:
        action = "checkin"

    now = datetime.now()
    date_str = request.GET.get("date") or now.strftime("%B %d, %Y")
    time_str = request.GET.get("time") or now.strftime("%I:%M:%S %p")

    # Keep name in session for the rest of the app.
    request.session["display_name"] = name

    role_key = "employee"
    if name != "Visitor":
        user, _ = RegisteredUser.objects.get_or_create(name=name)
        if user.account_role:
            role_key = user.account_role
    role_map = {
        "student": ("Student", "Access to courses"),
        "employee": ("Employee", "Standard portal access"),
        "teacher": ("Teacher", "Management tools"),
    }
    account_role_label, account_role_desc = role_map.get(role_key, role_map["employee"])

    return render(
        request,
        "attendance_success.html",
        {
            "name": name,
            "action": action,
            "date": date_str,
            "time": time_str,
            "account_role_label": account_role_label,
            "account_role_desc": account_role_desc,
        },
    )


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

        # Sign-in should ONLY verify identity (no attendance + no location/GPS checks).
        try:
            result = asyncio.run(system.recognize_image_bytes(image_bytes, mark_attendance=False))
        except Exception as exc:
            return JsonResponse({"success": False, "error": str(exc)}, status=500)

        if result.get("recognized"):
            # Provide a timestamp for UI display even when attendance isn't marked.
            ts = result.get("timestamp") or datetime.now().isoformat()
            # Persist name for dashboard greeting (no full auth yet).
            display_name = _normalize_display_name(result.get("name"))
            request.session["display_name"] = display_name
            return JsonResponse({
                "success": True,
                "name": display_name,
                "message": result.get("message", ""),
                "box": result.get("box"),
                "timestamp": ts,
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


def _extract_image_bytes_from_json(request: HttpRequest) -> tuple[bytes | None, JsonResponse | None]:
    """Parse a JSON payload containing an 'image' data-url/base64 field."""
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception as exc:
        return None, JsonResponse({"success": False, "error": f"Invalid JSON: {exc}"}, status=400)

    image_b64 = payload.get("image")
    if not image_b64:
        return None, JsonResponse({"success": False, "error": "Missing 'image' field"}, status=400)

    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        return base64.b64decode(image_b64), None
    except Exception as exc:
        return None, JsonResponse({"success": False, "error": f"Invalid base64 image: {exc}"}, status=400)


@require_POST
def attendance_checkin_scan(request: HttpRequest) -> JsonResponse:
    """Check-in: recognize face then record attendance (location rule applies)."""
    content_type = request.META.get("CONTENT_TYPE", "")
    if not content_type.startswith("application/json"):
        return JsonResponse({"success": False, "error": "Expected JSON request"}, status=400)

    image_bytes, err = _extract_image_bytes_from_json(request)
    if err:
        return err

    system = get_system()
    try:
        result = asyncio.run(system.recognize_image_bytes(image_bytes, mark_attendance=False))
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=500)

    if not result.get("recognized"):
        return JsonResponse(
            {"success": False, "message": result.get("message", "Face not recognized."), "box": result.get("box")},
            status=200,
        )

    name = _normalize_display_name(result.get("name"))
    request.session["display_name"] = name
    now = timezone.localtime(timezone.now())

    try:
        allowed, msg = asyncio.run(check_location_allowed())
    except Exception as exc:
        return JsonResponse({"success": False, "message": f"Location check failed: {exc}"}, status=500)

    if allowed:
        status = AttendanceRecord.STATUS_PRESENT
        time_value = now.time()
        message = "Check-in recorded."
    else:
        status = AttendanceRecord.STATUS_OUT_OF_RADIUS
        time_value = None
        message = msg or "Outside allowed radius."

    record = AttendanceRecord.objects.create(
        name=name,
        date=now.date(),
        time=time_value,
        status=status,
    )

    return JsonResponse(
        {
            "success": True,
            "name": record.name,
            "message": message,
            "timestamp": now.isoformat(),
        },
        status=200,
    )


@require_POST
def attendance_checkout_scan(request: HttpRequest) -> JsonResponse:
    """Check-out: recognize face then store check_out_time (location rule applies)."""
    content_type = request.META.get("CONTENT_TYPE", "")
    if not content_type.startswith("application/json"):
        return JsonResponse({"success": False, "error": "Expected JSON request"}, status=400)

    image_bytes, err = _extract_image_bytes_from_json(request)
    if err:
        return err

    system = get_system()
    try:
        result = asyncio.run(system.recognize_image_bytes(image_bytes, mark_attendance=False))
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=500)

    if not result.get("recognized"):
        return JsonResponse(
            {"success": False, "message": result.get("message", "Face not recognized."), "box": result.get("box")},
            status=200,
        )

    name = _normalize_display_name(result.get("name"))
    request.session["display_name"] = name
    now = timezone.localtime(timezone.now())

    try:
        allowed, msg = asyncio.run(check_location_allowed())
    except Exception as exc:
        return JsonResponse({"success": False, "message": f"Location check failed: {exc}"}, status=500)

    record = (
        AttendanceRecord.objects.filter(name=name, date=now.date())
        .order_by("-created_at")
        .first()
    )

    if record and record.check_out_time:
        return JsonResponse(
            {"success": False, "name": name, "message": "Already checked out for the latest session."},
            status=200,
        )

    if allowed:
        checkout_time = now.time()
        message = "Check-out recorded."
        status = AttendanceRecord.STATUS_PRESENT
    else:
        checkout_time = None
        message = msg or "Outside allowed radius."
        status = AttendanceRecord.STATUS_OUT_OF_RADIUS

    if record:
        record.check_out_time = checkout_time
        # If location fails at checkout, reflect it on the record as well.
        if status == AttendanceRecord.STATUS_OUT_OF_RADIUS:
            record.status = status
        record.save(update_fields=["check_out_time", "status"])
    else:
        record = AttendanceRecord.objects.create(
            name=name,
            date=now.date(),
            time=None,
            status=status,
            check_out_time=checkout_time,
        )

    return JsonResponse(
        {
            "success": True,
            "name": record.name,
            "message": message,
            "timestamp": now.isoformat(),
        },
        status=200,
    )


@require_POST
def register_face(request: HttpRequest) -> HttpResponse:
    """Register a student from the web UI. Supports form POST fallback and JSON camera capture."""
    if request.content_type and request.content_type.startswith("application/json"):
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except Exception as exc:
            return JsonResponse({"success": False, "error": f"Invalid JSON: {exc}"}, status=400)

        full_name = (payload.get("fullName") or "").strip()
        email = (payload.get("email") or "").strip()
        password = (payload.get("password") or "").strip()
        direction = (payload.get("direction") or "front").strip().lower()
        image_data = payload.get("image")
        if not full_name:
            return JsonResponse({"success": False, "error": "Please provide a full name."}, status=400)
        if not image_data:
            return JsonResponse({"success": False, "error": "Missing image data."}, status=400)

        allowed_dirs = {"front", "left", "right", "up", "down"}
        if direction not in allowed_dirs:
            direction = "front"

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as exc:
            return JsonResponse({"success": False, "error": f"Invalid image data: {exc}"}, status=400)

        try:
            from PIL import Image
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            return JsonResponse({"success": False, "error": f"Could not process image: {exc}"}, status=400)

        system = get_system()
        # Save as multi-angle references: sample_faces/<Name>/<direction>.jpg
        student_dir = os.path.join(system.database_path, full_name)
        os.makedirs(student_dir, exist_ok=True)
        save_path = os.path.join(student_dir, f"{direction}.jpg")

        try:
            img.save(save_path)
        except Exception as exc:
            return JsonResponse({"success": False, "error": f"Could not save image: {exc}"}, status=500)

        # Create/update user record.
        # New users should start with blank DOB/gender/account role; user fills later.
        try:
            user, created = RegisteredUser.objects.get_or_create(name=full_name)
            if created:
                user.dob_display = ""
                user.gender = ""
                user.account_role = ""
            if email:
                user.email = email
            if password:
                user.password = password
            user.save()
        except Exception:
            # Don't fail registration if DB write fails; face refs still saved.
            pass

        # Reload after DOWN capture (end of the guided flow)
        if direction == "down":
            try:
                system.load_database()
            except Exception:
                pass

        return JsonResponse(
            {
                "success": True,
                "message": f"Captured {direction}.",
            },
            status=200,
        )

    # Fallback for legacy form POST in case JavaScript is disabled.
    full_name = (request.POST.get("fullName") or "").strip()
    email = (request.POST.get("email") or "").strip()
    password = (request.POST.get("password") or "").strip()
    if not full_name:
        return render(
            request,
            "registration.html",
            {"error": "Please provide a name before registering."},
            status=400,
        )

    system = get_system()
    success = system.register_from_camera(full_name, camera_index=0)
    context = {
        "registration_success": success,
        "registered_name": full_name,
    }
    if not success:
        context["error"] = "Registration failed. Please try again in good lighting."
    else:
        try:
            user, created = RegisteredUser.objects.get_or_create(name=full_name)
            if created:
                user.dob_display = ""
                user.gender = ""
                user.account_role = ""
            if email:
                user.email = email
            if password:
                user.password = password
            user.save()
        except Exception:
            pass
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

