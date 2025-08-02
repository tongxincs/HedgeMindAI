
def format_box(lines: list[str], width: int = 80, padding: int = 2) -> str:
    """
    Formats a list of text lines into a centered, padded box of fixed width.

    Args:
        lines (list[str]): Lines of text to center inside the box.
        width (int): Total width of the box (including borders).
        padding (int): Spaces between border and text.

    Returns:
        str: A string representation of the formatted box.
    """
    box = []
    horizontal = "=" * width
    box.append(horizontal)

    for line in lines:
        content_width = width - 2 * padding - 2  # 2 for border pipes
        centered_line = line.center(content_width)
        box.append(f"{' ' * padding}|{centered_line}|")

    box.append(horizontal)
    return "\n".join(box)
