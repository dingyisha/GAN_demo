from curl_cffi import requests
import time
import random
import csv
cookies = {
    'machine_cookie': '5648673308365',
    '_pcid': '%7B%22browserId%22%3A%22lyshkhs0v10h46mc%22%7D',
    'pxcts': 'a098db9c-45af-11ef-a91d-3448701c5809',
    '_pxvid': 'a098d12d-45af-11ef-a91c-b245e34fedd7',
    '__pat': '-14400000',
    '_pctx': '%7Bu%7DN4IgrgzgpgThIC4B2YA2qA05owMoBcBDfSREQpAeyRCwgEt8oBJAE0RXQF8g',
    'LAST_VISITED_PAGE': '%7B%22pathname%22%3A%22https%3A%2F%2Fseekingalpha.com%2Fsymbol%2FAAPL%2Fnews%22%2C%22pageKey%22%3A%22aac40f29-0940-4e13-a81b-72005193bacb%22%7D',
    'userLocalData_mone_session_lastSession': '%7B%22machineCookie%22%3A5648673308365%2C%22machineCookieSessionId%22%3A%225648673308365%261721385408737%22%2C%22sessionStart%22%3A1721385408737%2C%22sessionEnd%22%3A1721387237435%2C%22firstSessionPageKey%22%3A%22a6313347-422c-4ec7-bff0-222e78741623%22%2C%22isSessionStart%22%3Afalse%2C%22lastEvent%22%3A%7B%22event_type%22%3A%22mousemove%22%2C%22timestamp%22%3A1721385437435%7D%7D',
    'sailthru_pageviews': '2',
    'sailthru_content': 'eaca8a3ebde49b34d28768f72fae972f',
    'sailthru_visitor': '68a422fa-e2c1-4cee-92a3-3a32d1fa01eb',
    '__pvi': 'eyJpZCI6InYtMjAyNC0wNy0xOS0xOC0zNi01MC00NzgtVW50SmRsWkR2bGxHV3Q3bC01ZWZkZjZiYWE3M2FlOWVlMmI1NzFiMzRkNzM0MTc0YiIsImRvbWFpbiI6Ii5zZWVraW5nYWxwaGEuY29tIiwidGltZSI6MTcyMTM4NTQzOTg5NX0%3D',
    '__tbc': '%7Bkpex%7DZjGpSbG6sgSehaIVrvqrRxnZRg-4VriordzMJL9axNswA75AfFmA_eSFgY7p3f_X',
    'xbc': '%7Bkpex%7Dtva_xVHLfMdgMGsLtaGxPg',
    '_px3': 'ad69a81e637536f1a0a383b28deaa649cf6d191410cfd5885280d04e88094d98:Tr0gPcZx1hhzRgh3BcFEh0bsHTwZIF3f7TSdVEnRc49krkEUBK9pTVFAuKdKR9qCzwqh4PBefw57h/4zasjaXQ==:1000:vLk8RbNbgNw9XZN1vnw0120kqjbi8viuJ9YV1aTDEXEfqJ+zAAqVR5K72vMBJhO195ar/7Ce2HQlRdgzaY8VjdEJzrIGrcQGtXPnWopdSx+4XbqJShc+QZS/C55ZIQEmQbw2ulX7o+mRLat8oeCiOy0+i2laLdtrfvznlGQBjC+XjwT8ZibooZ6HmwYdF5cCqS8RmP/jwK4olTmTa29kdkcHkjxd3MSADRPb+lrErIY=',
    '_pxde': '027cd5ccb3376ebe8787ead61085e659db8176e85475d9feb8212656e01fb3b3:eyJ0aW1lc3RhbXAiOjE3MjEzODU0NDM3MDAsImZfa2IiOjB9',
}

headers = {
    'authority': 'seekingalpha.com',
    'accept': '*/*',
    'accept-language': 'zh-CN,zh;q=0.9',
    'cache-control': 'no-cache',
    # 'cookie': 'machine_cookie=5648673308365; _pcid=%7B%22browserId%22%3A%22lyshkhs0v10h46mc%22%7D; pxcts=a098db9c-45af-11ef-a91d-3448701c5809; _pxvid=a098d12d-45af-11ef-a91c-b245e34fedd7; __pat=-14400000; _pctx=%7Bu%7DN4IgrgzgpgThIC4B2YA2qA05owMoBcBDfSREQpAeyRCwgEt8oBJAE0RXQF8g; LAST_VISITED_PAGE=%7B%22pathname%22%3A%22https%3A%2F%2Fseekingalpha.com%2Fsymbol%2FAAPL%2Fnews%22%2C%22pageKey%22%3A%22aac40f29-0940-4e13-a81b-72005193bacb%22%7D; userLocalData_mone_session_lastSession=%7B%22machineCookie%22%3A5648673308365%2C%22machineCookieSessionId%22%3A%225648673308365%261721385408737%22%2C%22sessionStart%22%3A1721385408737%2C%22sessionEnd%22%3A1721387237435%2C%22firstSessionPageKey%22%3A%22a6313347-422c-4ec7-bff0-222e78741623%22%2C%22isSessionStart%22%3Afalse%2C%22lastEvent%22%3A%7B%22event_type%22%3A%22mousemove%22%2C%22timestamp%22%3A1721385437435%7D%7D; sailthru_pageviews=2; sailthru_content=eaca8a3ebde49b34d28768f72fae972f; sailthru_visitor=68a422fa-e2c1-4cee-92a3-3a32d1fa01eb; __pvi=eyJpZCI6InYtMjAyNC0wNy0xOS0xOC0zNi01MC00NzgtVW50SmRsWkR2bGxHV3Q3bC01ZWZkZjZiYWE3M2FlOWVlMmI1NzFiMzRkNzM0MTc0YiIsImRvbWFpbiI6Ii5zZWVraW5nYWxwaGEuY29tIiwidGltZSI6MTcyMTM4NTQzOTg5NX0%3D; __tbc=%7Bkpex%7DZjGpSbG6sgSehaIVrvqrRxnZRg-4VriordzMJL9axNswA75AfFmA_eSFgY7p3f_X; xbc=%7Bkpex%7Dtva_xVHLfMdgMGsLtaGxPg; _px3=ad69a81e637536f1a0a383b28deaa649cf6d191410cfd5885280d04e88094d98:Tr0gPcZx1hhzRgh3BcFEh0bsHTwZIF3f7TSdVEnRc49krkEUBK9pTVFAuKdKR9qCzwqh4PBefw57h/4zasjaXQ==:1000:vLk8RbNbgNw9XZN1vnw0120kqjbi8viuJ9YV1aTDEXEfqJ+zAAqVR5K72vMBJhO195ar/7Ce2HQlRdgzaY8VjdEJzrIGrcQGtXPnWopdSx+4XbqJShc+QZS/C55ZIQEmQbw2ulX7o+mRLat8oeCiOy0+i2laLdtrfvznlGQBjC+XjwT8ZibooZ6HmwYdF5cCqS8RmP/jwK4olTmTa29kdkcHkjxd3MSADRPb+lrErIY=; _pxde=027cd5ccb3376ebe8787ead61085e659db8176e85475d9feb8212656e01fb3b3:eyJ0aW1lc3RhbXAiOjE3MjEzODU0NDM3MDAsImZfa2IiOjB9',
    'pragma': 'no-cache',
    'referer': 'https://seekingalpha.com/symbol/AAPL/news?from=2020-06-30T16%3A00%3A00.000Z&page=2&to=2024-07-01T15%3A59%3A59.999Z',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1',
}
for page in range(1,150):
    print(page)
    response = requests.get(
        f'https://seekingalpha.com/api/v3/symbols/aapl/news?filter[since]=1593576000&filter[until]=1719892799&id=aapl&include=author%2CprimaryTickers%2CsecondaryTickers%2Csentiments&isMounting=true&page[size]=0&page[number]={page}',
        cookies=cookies,
        headers=headers,
    )
    print(len(response.json()['data']))
    for i in response.json()['data']:
        title = i['attributes']['title']
        publishOn = i['attributes']['publishOn']
        with open('title.csv', 'a', encoding='utf-8-sig', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([title])
            with open('publishOn.csv', 'a', encoding='utf-8-sig', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([publishOn[:10]])
    if len(response.json()['data']) == 0:
        break
    time.sleep(random.uniform(1,2))