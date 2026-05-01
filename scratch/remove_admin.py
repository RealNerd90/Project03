import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from attendance.models import RegisteredUser, AdminAccount

def remove_admin():
    # 1. Remove "Admin" from RegisteredUser (User Management list)
    user_removed = RegisteredUser.objects.filter(name="Admin").delete()
    if user_removed[0] > 0:
        print(f"Successfully removed 'Admin' from RegisteredUser list.")
    else:
        print("No 'Admin' user found in RegisteredUser list.")

    # 2. Remove AdminAccount credentials
    admin_removed = AdminAccount.objects.filter(email="admin@gmail.com").delete()
    if admin_removed[0] > 0:
        print(f"Successfully removed 'admin@gmail.com' from AdminAccount credentials.")
    else:
        print("No 'admin@gmail.com' found in AdminAccount.")

if __name__ == "__main__":
    remove_admin()
