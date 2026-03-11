from django.urls import path

from . import views

urlpatterns = [
    path("", views.welcome, name="welcome"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("analytics/", views.analytics, name="analytics"),
    path("settings/", views.settings_view, name="settings"),
    path("scanner/", views.scanner, name="scanner"),
    path("signin/", views.signin, name="signin"),
    path("signin/scan/", views.signin_scan, name="signin-scan"),
    path("register/", views.registration, name="register"),
    path("register/face/", views.register_face, name="register-face"),
    path("api/attendance/", views.create_attendance_record, name="api-attendance-create"),
]

