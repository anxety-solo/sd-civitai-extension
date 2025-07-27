"""Logging utilities for the Civitai Extension."""

def log_message(
    message: str,
    status: str = 'info',
    verbose: bool = False
) -> None:
    """Display colored log messages with [CivitAi-Extension] prefix."""
    # Show messages only if verbose is True, except for errors and warnings
    if not verbose and status.lower() in ['info', 'success']:
        return

    # Define color codes for different status levels
    colors = {
        'error': '\033[31m',    # Red
        'warning': '\033[33m',  # Yellow
        'success': '\033[32m',  # Green
        'info': '\033[34m'      # Blue
    }

    # Get color for current status
    code_color = colors.get(status.lower(), '')
    reset = '\033[0m'
    tag = '[CivitAI-Extension]'

    # Format and print the message
    if code_color:
        print(f"{tag} - {code_color}{status.upper()}{reset} - {message}")
    else:
        print(f"{tag} - {status.upper()} - {message}")