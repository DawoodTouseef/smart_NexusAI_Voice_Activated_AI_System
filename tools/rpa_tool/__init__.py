import rpa as r
from Crewai.tools.base_tool import BaseTool
# Initialize RPA
r.init()

class open_website(BaseTool):
    def __init__(self):
        super().__init__( 
            name="Open Website",
            description="Open a specified website."
        )
    async def _run(self,url: str):
        """Open a specified website."""
        try:
            r.url(url)
            return f"Website '{url}' opened successfully."
        except Exception as e:
            return f"Failed to open website '{url}': {e}"

class click_element(BaseTool):
    def __init__(self):
        super().__init__(
            name="Click Element",
            description='Click an element specified by its selector.'
        )
    async def _run(self,selector: str):
        """Click an element specified by its selector."""
        try:
            r.click(selector)
            return f"Clicked element with selector '{selector}'."
        except Exception as e:
            return f"Failed to click element with selector '{selector}': {e}"

class type_text(BaseTool):
    def __init__(self):
        super().__init__(
            name="Type text",
            description='Type text into an input field specified by its selector.'
        )
    async def _run(self,selector: str, text: str):
        """Type text into an input field specified by its selector."""
        try:
            r.type(selector, text)
            return f"Typed text into element with selector '{selector}'."
        except Exception as e:
            return f"Failed to type text into element with selector '{selector}': {e}"

class take_screenshot(BaseTool):
    def __init__(self):
        super().__init__(
            name="Take ScreechShot",
            description='Take a screenshot and save it with the specified file name.'
        )
    async def _run(self,file_name: str):
        """Take a screenshot and save it with the specified file name."""
        try:
            r.snap('page', file_name)
            return f"Screenshot saved as '{file_name}'."
        except Exception as e:
            return f"Failed to take screenshot: {e}"

class get_clipboard(BaseTool):
    def __init__(self):
        super().__init__(
            name="Get Clipboard",
            description='Get the current content of the clipboard.'
        )
    async def _run(self):
        """Get the current content of the clipboard."""
        try:
            clipboard_content = r.clipboard()
            return f"Clipboard content: '{clipboard_content}'"
        except Exception as e:
            return f"Failed to get clipboard content: {e}"

class set_clipboard(BaseTool):
    def __init__(self):
        super().__init__(
            name='Set Clipboard',
            description='Set the clipboard content to the specified text.'
        )
    async def _run(self,text: str):
        """Set the clipboard content to the specified text."""
        try:
            r.clipboard(text)
            return f"Clipboard set to '{text}'."
        except Exception as e:
            return f"Failed to set clipboard content: {e}"
