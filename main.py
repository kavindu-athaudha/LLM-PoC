import asyncio
import math
import os
from typing import Dict, Any

import httpx
from httpx import Response
from sirius.ai.large_language_model import Conversation, LargeLanguageModel
from sirius.ai.open_ai import ChatGPTFunction


def get_instructions(instruction_id: int) -> Dict[str, str]:
    """
    Args:
        instruction_id: The instruction id for the instruction you want to retrieve. The instruction ID to retrieve financial advisors' instructions is 1. The instruction ID to retrieve a mechanic's instructions is 2.

    Returns:
        The instructions you should use to execute your next step
    """

    instruction_map: Dict[int, Dict] = {1: {"instructions": f"To recommend whether the user should invest in treasury bills or in retail sales, you should know the timeframe of the data the user wants you to analyse to come up with the conclusion."
                                                            f"If the user has not mentioned a timeframe, ask him to name a starting month and an ending month which are not more than 6 months apart and are within the last 1 year."
                                                            f"Next, retrieve the treasury yield data and the retail sales data using function calling. Use 'US' as the country code for both as we are only concerned with the United States."
                                                            f"Analyze both the data and isolate the data that fall within the starting month and ending month that the user stated."
                                                            f"If the treasury yield are generally rising within that time frame, recommend that investing in treasury bonds is a good investment and state reason why you think so."
                                                            f"If the treasury yield are generally decreasing within that time frame, recommend that investing in treasury bonds is a bad investment and state reason why you think so"
                                                            f"if the retail sales are generally rising within that time frame, recommend that investing in retail is a good investment and state reason why you think so."
                                                            f"If the retail sales are generally decreasing within that time frame, recommend that investing in retail sales is a bad investment and state reason why you think so."
                                                            f"It may be possible that both treasury yields and retail sales are a bad investment, both a good investment or one is a good investment while the other is a bad investment. If that is "
                                                            f"the case, reply stating that they are both good investments but since treasury bonds are backed by the government, they should invest in it rather than the retail sector."
                                                            f"Regardless of which investment is better, justify your answers with the data you used for the analysis. Always give a certain answer."}}
    return instruction_map.get(instruction_id)


def get_treasury_yield(country_code: str) -> Dict[str, Any]:
    """
    Args:
        country_code: The country code of the country where you want to get the treasury yield of

    Returns:
        The history of the past 1 year's treasury yield.
    """
    response: Response = httpx.get(f"https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=monthly&maturity=1year&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}")
    return response.json()


def get_retail_sales(country_code: str) -> Dict[str, Any]:
    """
    Args:
        country_code: The country code of the country where you want to get the retail sales of

    Returns:
        The history of the past 1 year's retail sales.
    """
    response: Response = httpx.get(f"https://www.alphavantage.co/query?function=RETAIL_SALES&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}")
    return response.json()


def buy_stock(stock_symbol: str, number_of_stocks: int, limit_price: float) -> Dict[str, Any]:
    """
    Args:
        stock_symbol: The stock trading symbol of the company you want to buy.
        number_of_stocks: The number of stocks you want to buy
        limit_price: The highest price you're willing to pay per stock

    Returns:
        Details about the trade and the stock portfolio after the trade
    """
    return {"orderStatus": "SUCCESS", "numberOfStockBought": number_of_stocks, "totalNumberOfStocksOwned": number_of_stocks + 200}


async def main():
    conversation: Conversation = Conversation.get_conversation(LargeLanguageModel.GPT4_TURBO, function_list=[ChatGPTFunction(get_instructions),
                                                                                                             ChatGPTFunction(get_treasury_yield),
                                                                                                             ChatGPTFunction(get_retail_sales),
                                                                                                             ChatGPTFunction(buy_stock)])
    conversation.add_system_prompt(f"You are an assistant that should only answer by retrieving the instructions needed to execute your next step."
                                   f"You can do only two things, it's either you can recommend whether the user should invest in treasury bills or in retail sales or the user can ask you to buy stocks."
                                   f"If the user wants you to recommend whether it's better to invest in treasury bills or in retail sales, you should retrieve the instructions with an instruction_id of 1."
                                   f"If the user wants you to buy stock, call the function that buys stocks and let the user know the number of stocks he bought, how many stocks he has in total and the stock code of the company.")

    while True:
        query: str = input("Ask: ")
        response: str = await conversation.say(query)
        # cost: float = (conversation.total_token_usage * 0.01 / 1000)
        print(f"Answer: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
