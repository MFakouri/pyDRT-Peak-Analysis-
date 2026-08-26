import click

@click.command()
def main():
    """Launch the pyDRTtools GUI."""
    from .GUI import launch_gui
    launch_gui()

if __name__ == '__main__':
    main()
