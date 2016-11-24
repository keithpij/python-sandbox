import urllib.request
import json
import models

def get_pricing(symbol):
    serviceurl = 'http://dev.markitondemand.com/MODApis/Api/v2/Quote/json?symbol='+symbol

    web = urllib.request.urlopen(serviceurl)
    data = web.read()
    web.close()

    print('Downloaded ' + str(len(data)) + ' characters.')

    try:
        js = json.loads(data.decode('utf-8'))
    except:
        js = None

    if js == None:
        print('Web Error')
    elif 'Status' not in js or js['Status'] != 'SUCCESS':
        print('==== Deserialization Error ====')
        print(data)
    else:
        price = models.Price()
        price.last_price = js['LastPrice']
        js['Timestamp']
        js['Open']
        js['Name']
        js['MarketCap']
        js['Change']
        js['Low']
        js['High']
        js['Volume']
        js['ChangePercentYTD']
        js['ChangePercent']
        js['Symbol']
        js['ChangeYTD']

        #print(json.dumps(js, indent=4))
