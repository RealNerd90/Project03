from django.db import models


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
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-date", "-time", "-created_at"]

    def __str__(self) -> str:
        return f"{self.name} - {self.date} {self.time or ''} ({self.status})"

