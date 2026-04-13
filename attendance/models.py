from django.db import models
from django.contrib.auth.hashers import check_password, make_password


class RegisteredUser(models.Model):
    """Basic registry of people known to the face-recognition system."""

    name = models.CharField(max_length=255, unique=True)
    email = models.EmailField(blank=True, default="")
    password = models.CharField(max_length=128, blank=True, default="")
    phone = models.CharField(max_length=50, blank=True, default="")
    dob_display = models.CharField(max_length=64, blank=True, default="")
    gender = models.CharField(max_length=16, blank=True, default="")
    account_role = models.CharField(max_length=16, blank=True, default="employee")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return f"{self.name} <{self.email or 'no-email'}>"


class AttendanceRecord(models.Model):
    """Single attendance event produced by the face-recognition system."""

    STATUS_PRESENT = "present"
    STATUS_OUT_OF_RADIUS = "out_of_radius"

    STATUS_CHOICES = [
        (STATUS_PRESENT, "Present"),
        (STATUS_OUT_OF_RADIUS, "Out of radius"),
    ]

    name = models.CharField(max_length=255)
    date = models.DateField()
    time = models.TimeField(null=True, blank=True)
    status = models.CharField(
        max_length=32,
        choices=STATUS_CHOICES,
        default=STATUS_PRESENT,
    )
    check_out_time = models.TimeField(null=True, blank=True)
    geofence = models.CharField(max_length=255, blank=True, default="Main Entrance")
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-date", "-time", "-created_at"]

    def __str__(self) -> str:
        return f"{self.name} - {self.date} {self.time or ''} ({self.status})"


class AdminAccount(models.Model):
    email = models.EmailField(unique=True)
    password_hash = models.CharField(max_length=256)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["email"]

    def __str__(self) -> str:
        return f"Admin <{self.email}>"

    def set_password(self, raw_password: str) -> None:
        self.password_hash = make_password(raw_password)

    def check_password(self, raw_password: str) -> bool:
        if not self.password_hash:
            return False
        return check_password(raw_password, self.password_hash)


class GeofenceSetting(models.Model):
    """Configuration for the site's geofence center and radius."""

    name = models.CharField(max_length=255, default="Main Site")
    latitude = models.FloatField(default=26.1180)  # Default to Guwahati
    longitude = models.FloatField(default=91.8136)
    radius = models.FloatField(default=500.0)
    verification_method = models.CharField(max_length=64, default="GPS Only")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Geofence Setting"
        verbose_name_plural = "Geofence Settings"

    def __str__(self) -> str:
        return f"{self.name} ({self.latitude}, {self.longitude}) - {self.radius}m"


class SystemSetting(models.Model):
    """Global configuration for the entire system."""

    default_language = models.CharField(max_length=255, default="English")
    timezone = models.CharField(max_length=255, default="Asia/Kolkata")
    maintenance_mode = models.BooleanField(default=False)
    retention_days = models.IntegerField(default=90)
    admin_email = models.EmailField(default="admin@gmail.com")
    reminder_interval = models.CharField(max_length=255, default="Daily Reminders")
    reminder_time = models.TimeField(default="08:30")
    enable_reminder_sound = models.BooleanField(default=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "System Setting"
        verbose_name_plural = "System Settings"

    def __str__(self) -> str:
        return f"System Config ({self.system_name})"

