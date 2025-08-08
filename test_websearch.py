from pprint import pprint as pp
from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun


def main():
    print("Running DuckDuckGo search tests...")
    test_duckduckgo_search()
    print("DuckDuckGo search test completed successfully.")


def test_duckduckgo_search():
    # Initialize the DuckDuckGo search tool
    search_tool = DuckDuckGoSearchResults()

    # Perform a search
    results = search_tool.run("Software engineering best practices")

    # Print the results
    pp(results)


if __name__ == "__main__":
    main()
