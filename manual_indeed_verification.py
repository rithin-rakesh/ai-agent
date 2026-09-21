import asyncio

from app.automation.playwright_manager import PlaywrightManager
from app.config.settings import get_settings

JOB_URL = "https://in.indeed.com/viewjob?jk=a415ab8e72d08fe0"

async def main():
    settings = get_settings()
    manager = PlaywrightManager(settings=settings)

    context = await manager.launch_persistent_context(
        user_data_dir=settings.get_indeed_profile_path(),
        headless=False,
    )

    page = context.pages[0] if context.pages else await context.new_page()

    try:
        await page.goto(
            JOB_URL,
            wait_until="domcontentloaded",
            timeout=settings.PLAYWRIGHT_TIMEOUT_MS,
        )

        print()
        print("Browser is open.")
        print("Complete any Indeed verification MANUALLY.")
        print("Do NOT click Apply.")
        print("When the actual job page is visible, come back here.")
        print("Then press ENTER.")
        print()

        input("Press ENTER after manual verification: ")

    finally:
        await manager.close_context(context)
        await manager.shutdown()

asyncio.run(main())
