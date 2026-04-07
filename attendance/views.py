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
        "static",
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
            "already_checked_in_today": today in present_days,
        },
    )


def admin_dashboard(request: HttpRequest) -> HttpResponse:
    """Admin dashboard UI with live system metrics."""
    if not request.session.get("is_admin"):
        return redirect("manual-login")

    # Read date range from query params, fallback to today
    start_date_str = request.GET.get("start_date")
    end_date_str = request.GET.get("end_date")
    
    today = timezone.localdate()
    try:
        from datetime import datetime
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date() if start_date_str else today
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date() if end_date_str else today
    except ValueError:
        start_date = today
        end_date = today

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    delta_days = (end_date - start_date).days + 1
    
    # Previous period for delta calculations
    prev_end = start_date - timedelta(days=1)
    prev_start = start_date - timedelta(days=delta_days)

    now = timezone.localtime(timezone.now())
    total_students = RegisteredUser.objects.count()

    # Selected date range's attendance percentage
    active_days_qs = AttendanceRecord.objects.filter(
        date__range=[start_date, end_date]
    ).values("date").distinct()
    active_days_count = active_days_qs.count() or 1 # avoid div zero
    
    present_range_count = AttendanceRecord.objects.filter(
        date__range=[start_date, end_date], status=AttendanceRecord.STATUS_PRESENT
    ).count()
    
    # average daily attendance % = (total present in range / (total_students * active_days)) * 100
    attendance_pct = (present_range_count / (total_students * active_days_count) * 100) if total_students > 0 else 0

    # Late arrivals (cutoff: 9:00 AM)
    late_cutoff = time_cls(9, 0)
    late_arrivals_count = AttendanceRecord.objects.filter(
        date__range=[start_date, end_date], time__gt=late_cutoff
    ).count()

    # Delta calculations (vs previous period)
    prev_active_days_qs = AttendanceRecord.objects.filter(
        date__range=[prev_start, prev_end]
    ).values("date").distinct()
    prev_active_days_count = prev_active_days_qs.count() or 1

    prev_present_count = AttendanceRecord.objects.filter(
        date__range=[prev_start, prev_end], status=AttendanceRecord.STATUS_PRESENT
    ).count()
    prev_attendance_pct = (prev_present_count / (total_students * prev_active_days_count) * 100) if total_students > 0 else 0
    
    attendance_delta = attendance_pct - prev_attendance_pct

    prev_late_count = AttendanceRecord.objects.filter(
        date__range=[prev_start, prev_end], time__gt=late_cutoff
    ).count()

    late_delta_pct = 0
    if prev_late_count > 0:
        late_delta_pct = ((late_arrivals_count - prev_late_count) / prev_late_count) * 100
    elif late_arrivals_count > 0:
        late_delta_pct = 100

    # ── Engagement Metrics ──────────────────────────────────────────────────
    # Active Now: incomplete sessions in the range
    active_now_count = AttendanceRecord.objects.filter(
        date__range=[start_date, end_date], status=AttendanceRecord.STATUS_PRESENT, check_out_time__isnull=True
    ).count()

    # New Today: students registered in the selected range
    new_today_count = RegisteredUser.objects.filter(created_at__date__range=[start_date, end_date]).count()

    # Peak Activity: hour with most check-ins in selected range
    range_records_with_time = list(AttendanceRecord.objects.filter(date__range=[start_date, end_date], time__isnull=False))
    if range_records_with_time:
        from collections import Counter
        hour_counter: Counter = Counter(r.time.hour for r in range_records_with_time)
        peak_hour = hour_counter.most_common(1)[0][0]
        peak_time_str = f"{peak_hour % 12 or 12}:00 {'AM' if peak_hour < 12 else 'PM'}"
    else:
        peak_time_str = "--"

    # Avg. Session: mean duration for sessions in range
    total_dur_secs = 0
    completed_sessions = 0
    for r in AttendanceRecord.objects.filter(date__range=[start_date, end_date], time__isnull=False, check_out_time__isnull=False):
        start_dt = datetime.combine(r.date, r.time)
        end_dt = datetime.combine(r.date, r.check_out_time)
        diff = (end_dt - start_dt).total_seconds()
        if diff > 0:
            total_dur_secs += diff
            completed_sessions += 1
    if completed_sessions > 0:
        avg_secs = int(total_dur_secs / completed_sessions)
        avg_session_str = f"{avg_secs // 60}m {avg_secs % 60}s"
    else:
        avg_session_str = "--"

    # Recent Logs (latest 5 records across all users in selected range)
    recent_records = AttendanceRecord.objects.filter(date__range=[start_date, end_date]).order_by("-date", "-time", "-created_at")[:5]
    logs = []
    for r in recent_records:
        logs.append({
            "name": r.name,
            "time": r.time.strftime("%I:%M %p").lstrip("0") if r.time else "--:--",
            "check_out_time": r.check_out_time.strftime("%I:%M %p").lstrip("0") if r.check_out_time else "--:--",
            "geofence": f"{r.latitude}, {r.longitude}" if r.latitude and r.longitude else (r.geofence or "Main Entrance"),
            "status": r.get_status_display().upper(),
            "status_class": "admin-badge--ok" if r.status == AttendanceRecord.STATUS_PRESENT else "admin-badge--warn",
        })

    # Map Markers (All records for today with coordinates)
    marker_records = AttendanceRecord.objects.filter(date__range=[start_date, end_date]).exclude(latitude__isnull=True).exclude(longitude__isnull=True)
    map_markers = []
    for r in marker_records:
        map_markers.append({
            "name": r.name,
            "lat": r.latitude,
            "lon": r.longitude,
            "status": r.get_status_display(),
            "color": "#10b981" if r.status == AttendanceRecord.STATUS_PRESENT else "#f59e0b",
        })

    # Geofence Setting for Map Center
    from .models import GeofenceSetting
    geofence_setting = GeofenceSetting.objects.first()
    if not geofence_setting:
        geofence_setting = GeofenceSetting.objects.create()

    context = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "map_markers": map_markers,
        "total_students": total_students,
        "attendance_pct": round(attendance_pct, 1),
        "attendance_delta": round(attendance_delta, 1),
        "late_arrivals": late_arrivals_count,
        "late_delta": round(late_delta_pct, 1),
        "active_now": active_now_count,
        "new_today": new_today_count,
        "peak_activity": peak_time_str,
        "avg_session": avg_session_str,
        "recent_logs": logs,
        "geofence_setting": geofence_setting,
        "active_page": "dashboard",
    }

    return render(request, "admin_dashboard.html", context)


def admin_geofencing(request: HttpRequest) -> HttpResponse:
    """Manage geofence settings (location center and radius)."""
    # In a real app, add @admin_required or similar check here
    from .models import GeofenceSetting
    
    setting = GeofenceSetting.objects.first()
    if not setting:
        setting = GeofenceSetting.objects.create() # Create default if missing

    if request.method == "POST":
        try:
            setting.latitude = float(request.POST.get("latitude", setting.latitude))
            setting.longitude = float(request.POST.get("longitude", setting.longitude))
            setting.radius = float(request.POST.get("radius", setting.radius))
            setting.verification_method = request.POST.get("verification_method", setting.verification_method)
            setting.save()
            messages.success(request, "Geofence settings updated successfully.")
            return redirect("admin-geofencing")
        except Exception as e:
            messages.error(request, f"Error updating settings: {e}")

    context = {
        "setting": setting,
        "active_page": "geofencing",
    }
    return render(request, "admin_geofencing.html", context)


def analytics(request: HttpRequest) -> HttpResponse:
    """Render the analytics page with real-time statistics and trends."""
    display_name = _normalize_display_name(request.session.get("display_name"))
    profile_photo_url = _profile_photo_url_for(display_name)
    
    if display_name == "Visitor":
        return redirect("signin")

    # Get date range from request or default to last 30 days
    today = timezone.localdate()
    default_start = today - timedelta(days=29)
    
    start_date_str = request.GET.get("start_date")
    end_date_str = request.GET.get("end_date")
    
    try:
        if start_date_str:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        else:
            start_date = default_start
            
        if end_date_str:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        else:
            end_date = today
    except ValueError:
        start_date = default_start
        end_date = today

    # Limit range to 30 days as requested
    if (end_date - start_date).days > 30:
        end_date = start_date + timedelta(days=30)

    # Fetch records for the user in the selected period
    records = AttendanceRecord.objects.filter(
        name=display_name,
        date__range=[start_date, end_date]
    ).order_by("date", "time")

    # Calculate Stats
    total_days = (end_date - start_date).days + 1
    
    # Identify working days (Mon-Fri)
    working_dates = []
    curr = start_date
    while curr <= end_date:
        if curr.weekday() < 5:  # 0-4 is Mon-Fri
            working_dates.append(curr)
        curr += timedelta(days=1)
    
    total_working_days = len(working_dates)
    present_dates = set(records.values_list("date", flat=True))
    present_count = len(present_dates)
    
    attendance_rate = (present_count / total_working_days * 100) if total_working_days > 0 else 0
    
    # Total Hours Worked
    total_seconds = 0
    for r in records:
        if r.time and r.check_out_time:
            # Simple duration within the same day
            dt1 = datetime.combine(r.date, r.time)
            dt2 = datetime.combine(r.date, r.check_out_time)
            if dt2 > dt1:
                total_seconds += (dt2 - dt1).total_seconds()
            else:
                # Handle overnight if necessary (though unlikely for this app)
                total_seconds += (dt2 + timedelta(days=1) - dt1).total_seconds()
    
    total_hours = total_seconds / 3600.0
    
    # Leaves (Gaps in working days)
    leaves_taken = total_working_days - present_count
    
    # Punctuality (Cutoff: 9:00 AM)
    ON_TIME_CUTOFF = time_cls(9, 0)
    total_checkins = records.filter(time__isnull=False).count()
    on_time_checkins = records.filter(time__lte=ON_TIME_CUTOFF).count()
    punctuality_score = (on_time_checkins / total_checkins * 100) if total_checkins > 0 else 0
    
    # Previous period for comparison (last 30 days before current range)
    prev_end = start_date - timedelta(days=1)
    prev_start = prev_end - timedelta(days=(end_date - start_date).days)
    prev_records = AttendanceRecord.objects.filter(
        name=display_name,
        date__range=[prev_start, prev_end]
    )
    prev_present_count = len(set(prev_records.values_list("date", flat=True)))
    
    # Calculate working days for previous period
    prev_working_count = 0
    curr = prev_start
    while curr <= prev_end:
        if curr.weekday() < 5:
            prev_working_count += 1
        curr += timedelta(days=1)
        
    prev_rate = (prev_present_count / prev_working_count * 100) if prev_working_count > 0 else 0
    rate_diff = attendance_rate - prev_rate

    # Trend Graph Data (Daily check-in times)
    trend_data = []
    curr = start_date
    while curr <= end_date:
        # Get first check-in of the day
        day_rec = records.filter(date=curr, time__isnull=False).order_by("time").first()
        if day_rec:
            # Convert time to decimal hours for the graph (e.g. 8:30 -> 8.5)
            h = day_rec.time.hour + (day_rec.time.minute / 60.0)
            trend_data.append({"date": curr.strftime("%b %d"), "hour": round(h, 2), "label": day_rec.time.strftime("%I:%M %p")})
        else:
            trend_data.append({"date": curr.strftime("%b %d"), "hour": None, "label": "No Data"})
        curr += timedelta(days=1)

    # Recent Activity
    activities = []
    # Get last 10 records for detailed activity
    recent_recs = AttendanceRecord.objects.filter(name=display_name).order_by("-date", "-created_at")[:10]
    for r in recent_recs:
        if r.time:
            is_on_time = r.time <= ON_TIME_CUTOFF
            activities.append({
                "kind": "checkin",
                "title": "Checked In",
                "time_label": f"{r.date.strftime('%b %d')} • {r.time.strftime('%I:%M %p')}",
                "meta": "On Time" if is_on_time else "Late Arrival",
                "icon": "ph-sign-in",
                "color_class": "bg-green-light text-green" if is_on_time else "bg-orange-light text-orange"
            })
        if r.check_out_time:
            # Calculate duration for this specific checkout
            duration_str = ""
            if r.time:
                dt1 = datetime.combine(r.date, r.time)
                dt2 = datetime.combine(r.date, r.check_out_time)
                diff = (dt2 - dt1).total_seconds() / 3600.0 if dt2 > dt1 else 0
                duration_str = f" • {diff:.1f}h Worked"
                
            activities.append({
                "kind": "checkout",
                "title": "Checked Out",
                "time_label": f"{r.date.strftime('%b %d')} • {r.check_out_time.strftime('%I:%M %p')}",
                "meta": f"Daily Session{duration_str}",
                "icon": "ph-sign-out",
                "color_class": "bg-blue-light text-primary"
            })

    # Streaks (Punctuality streak)
    streak_count = 0
    for r in recent_recs:
        if r.time and r.time <= ON_TIME_CUTOFF:
            streak_count += 1
        elif r.time:
            break

    # Progress bar percentages
    hours_goal = 160.0  # Default goal
    hours_pct = (total_hours / hours_goal * 100) if hours_goal > 0 else 0
    leaves_pct = (leaves_taken / 5 * 100) if 5 > 0 else 0 # Assuming 5 is a "high" number for visualization

    # 1. Monthly Summary (Current Year)
    monthly_summary = []
    m_year = today.year
    for m_month in range(1, 13):
        month_name = date(m_year, m_month, 1).strftime("%b").upper()
        
        # Monthly records
        m_records = AttendanceRecord.objects.filter(
            name=display_name,
            date__year=m_year,
            date__month=m_month
        )

        _, m_total_days = calendar.monthrange(m_year, m_month)
        
        # Monthly records - only first check-in of each day for punctuality
        m_day_records = m_records.filter(time__isnull=False).order_by("date", "time")
        m_first_checkins = {}
        for r in m_day_records:
            if r.date not in m_first_checkins:
                m_first_checkins[r.date] = r.time
        
        m_present_count = len(m_first_checkins)
        m_late_count = sum(1 for t in m_first_checkins.values() if t > ON_TIME_CUTOFF)
        m_absent_count = max(0, m_total_days - m_present_count)
        
        monthly_summary.append({
            "month": month_name,
            "present": m_present_count,
            "absent": m_absent_count,
            "late": m_late_count,
            "rate": (m_present_count / m_total_days * 100) if m_total_days > 0 else 0
        })

    # 2. Time Distribution (Selected Period)
    # Early: < 8:45 AM, On Time: 8:45-9:00 AM, Late: > 9:00 AM
    # Again, only first check-in of each day
    EARLY_CUTOFF = time_cls(8, 45)
    period_first_checkins = {}
    for r in records.filter(time__isnull=False).order_by("date", "time"):
        if r.date not in period_first_checkins:
            period_first_checkins[r.date] = r.time
            
    total_arrivals = len(period_first_checkins)
    early_count = sum(1 for t in period_first_checkins.values() if t < EARLY_CUTOFF)
    ontime_count = sum(1 for t in period_first_checkins.values() if EARLY_CUTOFF <= t <= ON_TIME_CUTOFF)
    late_count = sum(1 for t in period_first_checkins.values() if t > ON_TIME_CUTOFF)
    
    early_pct = (early_count / total_arrivals * 100) if total_arrivals > 0 else 0
    ontime_pct = (ontime_count / total_arrivals * 100) if total_arrivals > 0 else 0
    late_pct = (late_count / total_arrivals * 100) if total_arrivals > 0 else 0
    
    # Average Check-In Time & Duration
    if total_arrivals > 0:
        total_minutes = sum(t.hour * 60 + t.minute for t in period_first_checkins.values())
        avg_minutes = int(total_minutes / total_arrivals)
        avg_checkin_str = time_cls(avg_minutes // 60, avg_minutes % 60).strftime("%I:%M %p").lstrip("0")
    else:
        avg_checkin_str = "--:--"
        
    dur_count = sum(1 for r in records if r.time and r.check_out_time)
    avg_duration_str = f"{(total_hours / dur_count):.1f}" if dur_count > 0 else "0.0"
    
    # Efficiency is a mix of attendance and punctuality
    efficiency = (attendance_rate * 0.7 + punctuality_score * 0.3) if total_arrivals > 0 else attendance_rate

    # 3. Performance Insights
    most_consistent = max(monthly_summary, key=lambda x: x["rate"]) if monthly_summary else None
    
    # Longest Streak (All-time or at least last 100 records)
    all_recs = AttendanceRecord.objects.filter(name=display_name).order_by("date")
    max_streak = 0
    current_streak = 0
    last_date = None
    for r in all_recs:
        if last_date and (r.date - last_date).days == 1:
            current_streak += 1
        elif last_date and (r.date - last_date).days > 1:
            # Check if gap was only weekends
            is_weekend_gap = True
            check_date = last_date + timedelta(days=1)
            while check_date < r.date:
                if check_date.weekday() < 5:
                    is_weekend_gap = False
                    break
                check_date += timedelta(days=1)
            
            if is_weekend_gap:
                current_streak += 1
            else:
                max_streak = max(max_streak, current_streak)
                current_streak = 1
        else:
            current_streak = 1
        last_date = r.date
    max_streak = max(max_streak, current_streak)

    return render(
        request,
        "analytics.html",
        {
            "display_name": display_name,
            "profile_photo_url": profile_photo_url,
            "start_date": start_date,
            "end_date": end_date,
            "attendance_rate": round(attendance_rate, 1),
            "rate_diff": round(rate_diff, 1),
            "total_hours": round(total_hours, 1),
            "hours_pct": min(100, round(hours_pct, 1)),
            "leaves_taken": leaves_taken,
            "leaves_pct": min(100, round(leaves_pct, 1)),
            "punctuality_score": int(punctuality_score),
            "trend_data_json": json.dumps(trend_data),
            "recent_activities": activities[:5],
            "streak_count": streak_count,
            "attendance_records": records.order_by("-date", "-time")[:50],
            # New Data
            "current_year": today.year,
            "monthly_summary": monthly_summary,
            "early_pct": round(early_pct),
            "ontime_pct": round(ontime_pct),
            "late_pct": round(late_pct),
            "efficiency": round(efficiency),
            "avg_checkin_str": avg_checkin_str,
            "avg_duration_str": avg_duration_str,
            "most_consistent_month": most_consistent["month"] if most_consistent else "N/A",
            "most_consistent_rate": round(most_consistent["rate"]) if most_consistent else 0,
            "longest_streak": max_streak,
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
            "role_title": role_label or "Member",
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
        "email": email or "",
        "phone": phone or "",
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
    profile_photo_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "profile_photos")
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

    # Ensure single daily attendance
    existing_record = AttendanceRecord.objects.filter(
        name=name, date=now.date(), status=AttendanceRecord.STATUS_PRESENT
    ).exists()
    
    if existing_record:
        return JsonResponse(
            {
                "success": False,
                "already_checked_in": True,
                "message": "Attendance Already Recorded"
            },
            status=200,
        )

    try:
        allowed, msg, lat, lon = asyncio.run(check_location_allowed())
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
        geofence="Main Entrance", # default assigned
        latitude=lat,
        longitude=lon,
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
        allowed, msg, lat, lon = asyncio.run(check_location_allowed())
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
        record.latitude = lat
        record.longitude = lon
        # If location fails at checkout, reflect it on the record as well.
        if status == AttendanceRecord.STATUS_OUT_OF_RADIUS:
            record.status = status
        record.save()
    else:
        record = AttendanceRecord.objects.create(
            name=name,
            date=now.date(),
            time=None,
            status=status,
            check_out_time=checkout_time,
            geofence="Main Entrance",
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
        # Save as multi-angle references: media/<Name>/<direction>.jpg
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
            geofence=payload.get("geofence", "Main Entrance"),
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
