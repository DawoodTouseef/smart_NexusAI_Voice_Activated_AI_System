from Crewai import Task,Agent

class RPA_Tasks:
    def __init__(self,agent:Agent):
        super().__init__()
        self.rpa_agent=agent
    # Define RPA-specific tasks
    def open_website_task(self):
        return Task(
    description="Open a specified website.",
    expected_output="Website opened confirmation.",
    agent=self.rpa_agent,
    )

    def click_element_task(self):
        return Task(
    description="Click an element specified by its selector.",
    expected_output="Element clicked confirmation.",
    agent=self.rpa_agent,
    )

    def type_text_task(self):
        return Task(
    description="Type text into a specified input field.",
    expected_output="Text typed confirmation.",
    agent=self.rpa_agent,
    )

    def take_screenshot_task(self):
        return Task(
    description="Take a screenshot of the current page.",
    expected_output="Screenshot saved confirmation.",
    agent=self.rpa_agent,
    )

    def get_clipboard_task(self):
        return Task(
    description="Retrieve the current clipboard content.",
    expected_output="Clipboard content retrieved.",
    agent=self.rpa_agent,
    )

    def set_clipboard_task(self):
        return Task(
    description="Set the clipboard content to the specified text.",
    expected_output="Clipboard content set confirmation.",
    agent=self.rpa_agent,
    )