"""
Renova cookies TikTok via Playwright (IP residencial) e envia para o VPS.
Rodar periodicamente via Task Scheduler no PC do Renato.
"""
import asyncio
import time
import urllib.request


VPS_URL = "https://transcritor.demos.napel.com.br/api/cookies"


async def main():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            locale="pt-BR",
        )
        page = await ctx.new_page()
        await page.goto("https://www.tiktok.com/", wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(3000)

        cookies = await ctx.cookies()
        lines = ["# Netscape HTTP Cookie File", ""]
        for c in cookies:
            d = c["domain"]
            sub = "TRUE" if d.startswith(".") else "FALSE"
            sec = "TRUE" if c.get("secure") else "FALSE"
            exp = str(int(c.get("expires", time.time() + 365 * 86400)))
            lines.append(f"{d}\t{sub}\t{c.get('path','/')}\t{sec}\t{exp}\t{c['name']}\t{c['value']}")

        await browser.close()

    body = "\n".join(lines).encode()
    req = urllib.request.Request(VPS_URL, data=body, headers={"Content-Type": "text/plain"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        print(resp.read().decode())
    print(f"OK: {len(cookies)} cookies enviados ao VPS")


if __name__ == "__main__":
    asyncio.run(main())
