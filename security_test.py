import subprocess

def run_user_command(user_input: str) -> None:
    subprocess.run(user_input, shell=True, check=False)
