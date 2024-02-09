import requests
import json

def history_weather(location: str, date: str):
    KEY = "" # INPUT your key of weatherapi: http://api.weatherapi.com
    URL_HISTORY_WEATHER = "http://api.weatherapi.com/v1/history.json"

    # history
    param = {
                "key": KEY,
                "q": location,
                'dt': date,
                'hour': 12
            }
    res_completion = requests.get(URL_HISTORY_WEATHER, params=param)
    res_completion = json.loads(res_completion.text)
    output_dict = res_completion["forecast"]["forecastday"][0]['day']

    output = {}
    output["overall weather"] = f"{output_dict['condition']['text']}; "
    output["temperature"] = f"{output_dict['avgtemp_c']}(C); "
    output["wind speed"] = f"{output_dict['maxwind_kph']}(kph), {output_dict['maxwind_mph']}(mph); "
    output["precipitation"] = f"{output_dict['totalprecip_mm']}(mm), {output_dict['totalprecip_in']}(inch); "
    output["visibility"] = f"{output_dict['avgvis_km']}(km), {output_dict['avgvis_miles']}(miles); "
    output["humidity"] = f"{output_dict['avghumidity']}; "
    output["UV index"] = f"{output_dict['uv']}."

    format_output = {}
    format_output["overall weather"] = output_dict['condition']['text']
    format_output["temperature"] = output_dict['avgtemp_c']
    format_output["wind speed"] = output_dict['maxwind_kph']
    format_output["precipitation"] = output_dict['totalprecip_mm']
    format_output["visibility"] = output_dict['avgvis_km']
    format_output["humidity"] = output_dict['avghumidity']
    format_output["UV index"] = output_dict['uv']
    
    text_output = f"The weather for {param['q']} on {date} is: \n"+"".join([f"{key}: {output[key]}" for key in output.keys()])
    return text_output, format_output