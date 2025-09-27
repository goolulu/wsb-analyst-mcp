# mcp_wsb_server.py
import asyncio
import os
import re
import logging
import heapq
import time
import httpx

import asyncpraw
import aiohttp
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP, Context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Cache Configuration ---
CACHE_DATA = {}
CACHE_EXPIRY = {}
CACHE_TTL = 300  # Cache time-to-live in seconds (5 minutes)
# --- End Cache Configuration ---

mcp = FastMCP("WSB Analyst", dependencies=["asyncpraw", "pydantic", "httpx"])


class Comment(BaseModel):
    content: str
    score: int
    author: str


class Post(BaseModel):
    url: str
    title: str
    selftext: str
    upvote_ratio: float
    link_flair_text: str
    top_comments: list[Comment] = Field(default_factory=list)
    extracted_links: list[str] = Field(default_factory=list)


# ---- Helper functions ----


async def get_reddit_client():
    try:
        client_id = os.environ.get("REDDIT_CLIENT_ID")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET")

        if not client_id or not client_secret:
            logger.error("Reddit API credentials not found in environment variables")
            return None
        proxy_url = "http://127.0.0.1:1080"
        session = aiohttp.ClientSession(proxy=proxy_url)
        return asyncpraw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="cherry studio:v1.0.0 (by /u/Abellettor)",
            requestor_kwargs={"session": session},
        )
    except Exception as e:
        logger.error(f"Error creating Reddit client: {str(e)}")
        return None


def is_valid_external_link(url: str) -> bool:
    excluded_domains = [
        "reddit.com",
        "redd.it",
        "imgur.com",
        "gfycat.com",
        "redgifs.com",
        "giphy.com",
        "imgflip.com",
        "youtu.be",
        "discord.gg",
    ]
    if any(domain in url for domain in excluded_domains):
        return False

    return True


def extract_valid_links(text: str) -> list[str]:
    if not text:
        return []
    url_pattern = re.compile(
        r"https?://(?!(?:www\.)?reddit\.com|(?:www\.)?i\.redd\.it|(?:www\.)?v\.redd\.it|(?:www\.)?imgur\.com|"
        r"(?:www\.)?preview\.redd\.it|(?:www\.)?sh\.reddit\.com|[^.]*\.reddit\.com)"
        r'[^\s)\]}"\']+',
        re.IGNORECASE,
    )
    links = url_pattern.findall(text)
    return [link for link in links if is_valid_external_link(link)]


async def filter_valid_tickers(tickers: list[str]) -> list[str]:
    """Filter out invalid tickers using NASDAQ's symbol list."""
    async with httpx.AsyncClient() as client:
        try:
            nasdaq_response = await client.get(
                "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
            )
            other_response = await client.get(
                "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
            )
            nasdaq_response.raise_for_status()
            other_response.raise_for_status()
        except httpx.RequestError as e:
            logger.error(f"Error fetching NASDAQ ticker lists: {e}")
            return list(set(ticker.upper() for ticker in tickers))
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Error fetching NASDAQ ticker lists (status {e.response.status_code}): {e}"
            )
            return list(set(ticker.upper() for ticker in tickers))

        valid_symbols = set()
        try:
            for line in nasdaq_response.text.splitlines()[
                1:-1
            ]:  # Skip header and footer
                parts = line.split("|")
                if len(parts) > 0:
                    symbol = parts[0]
                    valid_symbols.add(symbol)

            for line in other_response.text.splitlines()[
                1:-1
            ]:  # Skip header and footer
                parts = line.split("|")
                if len(parts) > 0:
                    symbol = parts[0]
                    valid_symbols.add(symbol)
        except Exception as e:
            logger.error(f"Error parsing NASDAQ ticker lists: {e}")
            return list(set(ticker.upper() for ticker in tickers))

    unique_tickers = set(ticker.upper() for ticker in tickers)
    return [ticker for ticker in unique_tickers if ticker in valid_symbols]


# ---- MCP Tools ----


@mcp.tool()
async def find_top_posts(
    min_score: int = 100,
    min_comments: int = 10,
    limit: int = 10,
    excluded_flairs: list[str] = ["Meme", "Shitpost", "Gain", "Loss"],
    ctx: Context = None,
) -> dict:
    """
    Fetch and filter WSB posts based on criteria. Caches results for 5 minutes.

    Args:
        min_score: Minimum score (upvotes) required
        min_comments: Minimum number of comments required
        limit: Maximum number of posts to return
        excluded_flairs: List of post flairs to exclude. Defaults to ["Meme", "Shitpost", "Gain", "Loss"].

    Returns:
        A dictionary with filtered posts data
    """
    # --- Cache Check ---
    # Sort flairs to ensure consistent key regardless of order
    excluded_flairs_tuple = tuple(sorted(excluded_flairs))
    cache_key = (
        f"find_top_posts:{min_score}:{min_comments}:{limit}:{excluded_flairs_tuple}"
    )
    current_time = time.time()
    if cache_key in CACHE_DATA and current_time < CACHE_EXPIRY.get(cache_key, 0):
        logger.info(f"Cache hit for {cache_key}")
        return CACHE_DATA[cache_key]
    logger.info(f"Cache miss for {cache_key}")
    # --- End Cache Check ---

    try:
        if ctx:
            await ctx.report_progress(0, 2)

        reddit = await get_reddit_client()
        if not reddit:
            return {"error": "Unable to connect to Reddit API. Check your credentials."}

        try:
            # Fetch posts
            if ctx:
                await ctx.report_progress(1, 2)

            subreddit = await reddit.subreddit("wallstreetbets")
            hot_posts = subreddit.hot(limit=50)

            top_posts_heap = []  # Min-heap storing (score, post_dict)

            async for post in hot_posts:
                # Filter
                if (
                    post.score >= min_score
                    and post.num_comments >= min_comments
                    and (post.link_flair_text or "") not in excluded_flairs
                ):
                    post_data = {
                        "id": post.id,
                        "url": f"https://www.reddit.com{post.permalink}",
                        "title": post.title,
                        "selftext": post.selftext,
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "upvote_ratio": post.upvote_ratio,
                        "link_flair_text": post.link_flair_text or "",
                        "created_utc": post.created_utc,
                    }

                    if len(top_posts_heap) < limit:
                        heapq.heappush(top_posts_heap, (post.score, post_data))
                    elif (
                        post.score > top_posts_heap[0][0]
                    ):  # Compare with min score in heap
                        # If current post is better than the worst in the heap, replace it
                        heapq.heapreplace(top_posts_heap, (post.score, post_data))

            # Extract posts from heap and sort descending by score
            # The heap contains the top 'limit' posts based on score, but not necessarily sorted
            top_posts = sorted(
                [item[1] for item in top_posts_heap],
                key=lambda x: x["score"],
                reverse=True,
            )

            logger.info(
                f"Processed posts, selected top {len(top_posts)} posts meeting criteria"
            )

            if ctx:
                await ctx.report_progress(2, 2)

            result = {"count": len(top_posts), "posts": top_posts}

            # --- Cache Store ---
            CACHE_DATA[cache_key] = result
            CACHE_EXPIRY[cache_key] = current_time + CACHE_TTL
            logger.info(f"Cached result for {cache_key} with TTL {CACHE_TTL}s")
            # --- End Cache Store ---

            return result
        finally:
            await reddit.close()
    except Exception as e:
        logger.error(f"Error in fetch_posts: {str(e)}")
        return {"error": f"Failed to fetch posts: {str(e)}"}


@mcp.tool()
async def fetch_post_details(post_id: str, ctx: Context = None) -> dict:
    """
    Fetch detailed information about a specific WSB post including top comments. Caches results for 5 minutes.

    Args:
        post_id: Reddit post ID

    Returns:
        Detailed post data including comments and extracted links
    """
    # --- Cache Check ---
    cache_key = f"fetch_post_details:{post_id}"
    current_time = time.time()
    if cache_key in CACHE_DATA and current_time < CACHE_EXPIRY.get(cache_key, 0):
        logger.info(f"Cache hit for {cache_key}")
        return CACHE_DATA[cache_key]
    logger.info(f"Cache miss for {cache_key}")
    # --- End Cache Check ---

    try:
        if ctx:
            await ctx.report_progress(0, 3)

        reddit = await get_reddit_client()
        if not reddit:
            return {"error": "Unable to connect to Reddit API. Check your credentials."}

        try:
            if ctx:
                await ctx.report_progress(1, 3)

            submission = await reddit.submission(id=post_id)

            # Load comments
            if ctx:
                await ctx.report_progress(2, 3)

            await submission.comments.replace_more(limit=0)
            comments = await submission.comments.list()
            top_comments = sorted(comments, key=lambda c: c.score, reverse=True)[:10]

            # Extract links
            content_links = []
            if not submission.is_self and is_valid_external_link(submission.url):
                content_links.append(submission.url)
            elif submission.is_self:
                content_links.extend(extract_valid_links(submission.selftext))

            # Process comments
            comment_links = []
            comment_data = []
            for comment in top_comments:
                try:
                    author_name = comment.author.name if comment.author else "[deleted]"
                    links_in_comment = extract_valid_links(comment.body)
                    if links_in_comment:
                        comment_links.extend(links_in_comment)

                    comment_data.append(
                        {
                            "content": comment.body,
                            "score": comment.score,
                            "author": author_name,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error processing comment: {str(e)}")

            # Combine all found links
            all_links = list(set(content_links + comment_links))

            result = {
                "post_id": post_id,
                "url": f"https://www.reddit.com{submission.permalink}",
                "title": submission.title,
                "selftext": submission.selftext if submission.is_self else "",
                "upvote_ratio": submission.upvote_ratio,
                "score": submission.score,
                "link_flair_text": submission.link_flair_text or "",
                "top_comments": comment_data,
                "extracted_links": all_links,
            }

            # --- Cache Store ---
            CACHE_DATA[cache_key] = result
            CACHE_EXPIRY[cache_key] = current_time + CACHE_TTL
            logger.info(f"Cached result for {cache_key} with TTL {CACHE_TTL}s")
            # --- End Cache Store ---

            if ctx:
                await ctx.report_progress(3, 3)

            return result
        finally:
            await reddit.close()
    except Exception as e:
        logger.error(f"Error in fetch_post_details: {str(e)}")
        return {"error": f"Failed to fetch post details: {str(e)}"}


@mcp.tool()
async def fetch_batch_post_details(post_ids: list[str], ctx: Context = None) -> dict:
    """
    Fetch details for multiple posts efficiently.

    Args:
        post_ids: List of Reddit post IDs

    Returns:
        Dictionary with details for all requested posts
    """
    if not post_ids:
        return {"error": "No post IDs provided"}

    results = {}
    total = len(post_ids)

    for i, post_id in enumerate(post_ids):
        if ctx:
            await ctx.report_progress(i, total)

        detail = await fetch_post_details(post_id)
        results[post_id] = detail

    if ctx:
        await ctx.report_progress(total, total)

    return {"count": len(results), "posts": results}


@mcp.tool()
async def fetch_detailed_wsb_posts(
    min_score: int = 100,
    min_comments: int = 10,
    limit: int = 10,
    excluded_flairs: list[str] = ["Meme", "Shitpost", "Gain", "Loss"],
    ctx: Context = None,
) -> dict:
    """
    Fetch and filter WSB posts, then get detailed information including top comments and links for each.

    Args:
        min_score: Minimum score (upvotes) required
        min_comments: Minimum number of comments required
        limit: Maximum number of posts to return
        excluded_flairs: List of post flairs to exclude. Defaults to ["Meme", "Shitpost", "Gain", "Loss"].

    Returns:
        A dictionary with detailed data for the filtered posts.
    """
    if ctx:
        await ctx.report_progress(0, 3)

    # Step 1: Fetch initial posts based on criteria
    posts_result = await find_top_posts(
        min_score=min_score,
        min_comments=min_comments,
        limit=limit,
        excluded_flairs=excluded_flairs,
        ctx=None,  # Don't pass context down, manage progress here
    )

    if "error" in posts_result:
        logger.error(
            f"Error during initial post fetch in fetch_detailed_wsb_posts: {posts_result['error']}"
        )
        if ctx:
            await ctx.report_progress(3, 3)
        return {"error": f"Failed during initial post fetch: {posts_result['error']}"}

    if not posts_result["posts"]:
        logger.info("No posts found matching criteria in fetch_detailed_wsb_posts.")
        if ctx:
            await ctx.report_progress(3, 3)
        return {"count": 0, "posts": {}}

    post_ids = [post["id"] for post in posts_result["posts"]]
    logger.info(f"Found {len(post_ids)} posts matching criteria, fetching details...")

    if ctx:
        await ctx.report_progress(1, 3)

    # Step 2: Fetch detailed information for the filtered posts
    # Pass the context down to fetch_batch_post_details for finer-grained progress within that step
    details_result = await fetch_batch_post_details(
        post_ids=post_ids, ctx=ctx
    )  # Pass context here

    if "error" in details_result:
        logger.error(
            f"Error during batch detail fetch in fetch_detailed_wsb_posts: {details_result['error']}"
        )
        # Progress reporting is handled within fetch_batch_post_details if ctx is passed
        return {"error": f"Failed during batch detail fetch: {details_result['error']}"}

    # Progress reporting completion is handled within fetch_batch_post_details
    logger.info(
        f"Successfully fetched details for {len(details_result.get('posts', {}))} posts."
    )

    return details_result  # Return the structure from fetch_batch_post_details


@mcp.tool()
async def get_top_trending_tickers(
    num_stocks: int = 20, filter: str = "wallstreetbets"
) -> list[str]:
    """
    Fetch top trending stock tickers from ApeWisdom, filtered by valid NASDAQ symbols.

    Args:
        num_stocks: Number of top stocks to consider based on upvotes and mentions. Defaults to 20.
        filter: ApeWisdom filter category (e.g., 'wallstreetbets', 'all'). Defaults to 'wallstreetbets'.

    Returns:
        A list of valid, trending tickers.
    """
    api_url = f"https://apewisdom.io/api/v1.0/filter/{filter}"
    logger.info(f"Fetching trending tickers from {api_url}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(api_url)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()
            logger.info(
                f"Successfully fetched data from ApeWisdom for filter '{filter}'"
            )
        except httpx.RequestError as e:
            logger.error(f"Error fetching from ApeWisdom: {e}")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"ApeWisdom API returned error {e.response.status_code}: {e}")
            return []
        except Exception as e:  # Catch potential JSON decoding errors etc.
            logger.error(
                f"An unexpected error occurred fetching/parsing ApeWisdom data: {e}"
            )
            return []

    if "results" not in data or not isinstance(data["results"], list):
        logger.error("Invalid data structure received from ApeWisdom API.")
        return []

    potential_tickers = set()
    metrics = ["upvotes", "mentions"]
    for metric in metrics:
        # Sort by the current metric and take top N
        # Add try-except for robustness against missing keys or non-numeric values
        try:
            sorted_stocks = sorted(
                [
                    stock
                    for stock in data["results"]
                    if isinstance(stock.get(metric), (int, float))
                ],
                key=lambda item: item.get(metric, 0),
                reverse=True,
            )
            for stock in sorted_stocks[:num_stocks]:
                if "ticker" in stock:
                    potential_tickers.add(stock["ticker"])
        except Exception as e:
            logger.warning(f"Could not process metric '{metric}' due to error: {e}")

    if not potential_tickers:
        logger.info("No potential tickers found after processing ApeWisdom results.")
        return []

    logger.info(f"Found {len(potential_tickers)} potential tickers, filtering...")

    # Filter valid tickers
    valid_tickers = await filter_valid_tickers(list(potential_tickers))
    logger.info(f"Filtered down to {len(valid_tickers)} valid trending tickers.")

    return valid_tickers


@mcp.tool()
async def get_external_links(
    min_score: int = 100, min_comments: int = 10, limit: int = 10, ctx: Context = None
) -> dict:
    """
    Get all external links from top WSB posts.

    Args:
        min_score: Minimum score (upvotes) required
        min_comments: Minimum number of comments required
        limit: Maximum number of posts to scan

    Returns:
        Dictionary with all unique external links found
    """
    if ctx:
        await ctx.report_progress(0, 3)

    # Get filtered posts
    posts_result = await find_top_posts(min_score, min_comments, limit)
    if "error" in posts_result:
        return {"error": posts_result["error"]}

    if len(posts_result["posts"]) == 0:
        return {"count": 0, "links": []}

    # Collect post IDs
    post_ids = [post["id"] for post in posts_result["posts"]]

    if ctx:
        await ctx.report_progress(1, 3)

    # Get details for all posts
    details_result = await fetch_batch_post_details(post_ids)
    if "error" in details_result:
        return {"error": details_result["error"]}

    # Extract all links
    all_links = []
    for post_id, post_detail in details_result["posts"].items():
        if "extracted_links" in post_detail:
            all_links.extend(post_detail["extracted_links"])

    if ctx:
        await ctx.report_progress(2, 3)

    # Remove duplicates and sort
    unique_links = sorted(list(set(all_links)))

    if ctx:
        await ctx.report_progress(3, 3)

    return {"count": len(unique_links), "links": unique_links}


# ---- MCP Prompts ----


@mcp.prompt()
def analyze_wsb_market() -> str:
    """Create a prompt for analyzing WSB market sentiment and finding opportunities."""

    return """Analyze these WallStreetBets posts to identify market opportunities and sentiment.

Use the fetch_detailed_wsb_posts tool to get the complete content of filtered posts.

First, review the data to understand what retail investors are discussing.
Then, create a market analysis report with the following sections:

1. **Title:** Witty and thematic
2. **Key Market Opportunities (2-3 bullets):** Highlight promising opportunities with specific stock tickers
3. **Analysis (3-5 paragraphs):** Synthesize findings on market trends, include bold takes (backed by evidence)
4. **Potential Options Plays (1-2 recommendations):** Suggest specific options strategies/contracts
5. **Conclusion:** Punchy summary

Guidelines:
- Be concise, witty, and data-driven
- Focus on market opportunities and catalysts
- Include specific tickers when relevant
- Be objective and analytical

Additional instructions:
- Examine external links shared in posts using the get_external_links tool
- Consider what information sources retail investors are valuing
"""


@mcp.prompt()
def find_market_movers(ticker: str = "") -> str:
    """
    Create a prompt for identifying what's moving a specific stock or the market.

    Args:
        ticker: Optional specific stock ticker to focus on
    """
    if ticker:
        prompt = f"""Analyze WallStreetBets discussions about ${ticker} to understand what's driving its price movement.

Use the find_top_posts tool to get recent posts, looking specifically for mentions of ${ticker}.
Then use fetch_post_details to get detailed content of relevant posts.

In your analysis:
1. Identify key catalysts being discussed
2. Summarize the prevailing sentiment (bullish/bearish)
3. Note any upcoming events that might impact the stock
4. List any external links being shared about this ticker
"""
    else:
        prompt = """Analyze WallStreetBets to identify which stocks are seeing unusual attention and why.

Use the find_top_posts tool to get recent popular posts.
Then use fetch_batch_post_details to gather more information.

In your analysis:
1. Identify the top 3-5 stocks being discussed most actively
2. For each, summarize the catalysts driving attention
3. Note the sentiment direction for each (bullish/bearish)
4. Highlight any unusual patterns in the discussions
"""

    return prompt


def main():
    mcp.run(transport="stdio")


# Run the server
if __name__ == "__main__":
    os.environ["REDDIT_CLIENT_ID"] = "he4dCnWhWy2mTBrcPCb02g"
    os.environ["REDDIT_CLIENT_SECRET"] = "KANViIYhwDUZ5qimP3fzEjQm8yHGiw"
    result = asyncio.run(
        find_top_posts(100, 10, 1, ["Meme", "Shitpost", "Gain", "Loss"], None)
    )

    print(result)
