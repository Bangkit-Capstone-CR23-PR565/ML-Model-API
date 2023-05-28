
import csv
from bs4 import BeautifulSoup
import pandas as pd
import requests

def fetch_events():
    # Choose one
    def fetch_online_file():
        response = requests.get("https://cdn.discordapp.com/attachments/1106183330547892234/1110213188412243998/indonesia.html")
        return response.content
    
    def fetch_local_file():
        with open("./assets/Scrapped_site.html", "r") as file:
            return file.read()
    
    response_content = fetch_online_file()
    soup = BeautifulSoup(response_content, 'html.parser')

    table = soup.find('table')

    # headers = [header.text.strip() for header in table.find_all('th')]
    headers = ['id', 'date', 'event_name', 'location', 'description', 'tags', 'interested']

    rows = []
    id = 1
    for tr in table.find_all('tr'):
        row = []

        # pass featured cards
        if (tr.find('svg', class_="svg-inline--fa fa-star fa-w-18 me-1 text-orange-l") != None):
            continue

        row.append(str(id))

        data = tr.find('div', class_='small fw-500')
        row.append(data.text) if data else row.append('')

        data = tr.find('span', class_='d-block')
        row.append(data.text.strip()) if data else row.append('')

        data = tr.find('div', class_='small fw-500 venue')
        row.append(data.text.replace('\xa0•\xa0Online', '')) if data else row.append('')

        data = tr.find('div', class_='small text-wrap text-break')
        row.append(data.text) if data else row.append('')

        # not enough data
        if (len(''.join(row[1:5]).strip()) == 0):
            continue

        data = tr.find('td', class_='col-12 small text-muted mb-2')
        row.append([tag.text.strip() for tag in data if tag.text.strip() != '']) if data else row.append('')

        data = tr.find('a', class_='small fw-500 text-decoration-none px-2 xn')
        row.append(data.text) if data else row.append('')

        rows.append(row)

        id += 1

    # Specify the CSV file path
    csv_file = 'events.csv'

    # Write the data to the CSV file
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

def fetch_ratings():
    key='9aadb790'
    user_count=1000
    generate_count=1000
    event_count=400

    ratings_df = pd.read_csv(f"https://my.api.mockaroo.com/rating.json?key={key}&count={generate_count}&user_id={user_count}&event_id={event_count}")

    # drop duplicated
    duplicated_series = ratings_df.duplicated(subset=['user_id', 'event_id'], keep=False)
    for row in range(0,len(duplicated_series)):
        if duplicated_series[row] == True:
            ratings_df.drop(index=row, inplace=True)

    ratings_df.reset_index(drop=True, inplace=True)

    # check for duplicated
    duplicated_series = ratings_df.duplicated(subset=['user_id', 'event_id'], keep=False)
    for row in range(0,len(duplicated_series)):
        if duplicated_series[row] == True:
            print(row)

    ratings_df.to_csv('ratings.csv', sep=',', encoding='utf-8')