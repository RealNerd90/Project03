from django.urls import path # type: ignore

from . import views # type: ignore

urlpatterns = [
    path("", views.welcome, name="welcome"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("admin-dashboard/", views.admin_dashboard, name="admin-dashboard"),
    path("admin-dashboard/geofencing/", views.admin_geofencing, name="admin-geofencing"),
    path("admin-dashboard/users/", views.admin_user_management, name="admin-user-management"),
    path("admin-dashboard/enroll/", views.admin_enroll_user, name="admin-enroll-user"),
    path("admin-dashboard/settings/", views.admin_system_settings, name="admin-system-settings"),
    path("analytics/", views.analytics, name="analytics"),
    path("profile/", views.profile_view, name="profile"),
    path("profile/edit/", views.profile_edit_view, name="profile-edit"),
    path("settings/", views.settings_view, name="settings"),
    path("scanner/", views.scanner, name="scanner"),
    path("signin/", views.signin, name="signin"),
    path("login/", views.manual_login, name="manual-login"),
    path("signin/scan/", views.signin_scan, name="signin-scan"),
    path("signin/success/", views.signin_success, name="signin-success"),
    path("logout/", views.logout_view, name="logout"),
    path("attendance/success/", views.attendance_success, name="attendance-success"),
    path("attendance/checkin/scan/", views.attendance_checkin_scan, name="attendance-checkin-scan"),
    path("attendance/checkout/scan/", views.attendance_checkout_scan, name="attendance-checkout-scan"),
    path("register/", views.registration, name="register"),
    path("register/face/", views.register_face, name="register-face"),
    path("api/attendance/", views.create_attendance_record, name="api-attendance-create"),
]

