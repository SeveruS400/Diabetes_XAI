# This is a sample Python client for Replica API.
# Available at https://gitlab.com/snippets/1960032
# See full API docs: https://docs.replicastudios.com/
import base64
import json
import time
import traceback
import requests

root_url = 'https://api.replicastudios.com'
client_id = 'mahirfurkan1999@gmail.com'
secret = 'Deneme153426'

#txt = 'Hello world'
audio_format = 'wav'

# f = open("demofile2.txt", "a")
# f.write("Now the file has more content!")
# f.close()
#
# #open and read the file after the appending:
# f = open("demofile2.txt", "r")
# print(f.read())
def get_access_token(client_id, secret):
    """
    Authenticates in /auth endpoint and returns the access token string.
    Note: token expires after an hour.
    """
    url = f"{root_url}/auth"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = f'client_id={client_id}&secret={secret}'
    response = requests.request("POST", url,
                                headers=headers,
                                data=payload)
    if response.status_code != 200:
        raise Exception(f'Auth failed. Response: {response.text}')
    # extract access_token from response
    jwt_access_token = response.json()['access_token']
    print(f'Acquired JWT token: {jwt_access_token}\n')
    return jwt_access_token


def decode_jwt(access_token):
    """Takes raw JWT and returns the content of payload part as dictionary."""
    payload_b64 = access_token.split('.')[1]
    # fix base64 padding
    payload_b64 = payload_b64 + '=' * (4 - len(payload_b64) % 4)
    json_payload = base64.urlsafe_b64decode(payload_b64)
    payload_dict = json.loads(json_payload)
    return payload_dict


def get_voices(access_token):
    """Requests list of voices from /voice endpoint."""
    url = f"{root_url}/voice"
    response = requests.request("GET", url, headers={'Authorization': f'Bearer {access_token}'}, data={})
    voices = response.json()
    print(f'Found {len(voices)} voices.')
    return voices


def text_to_speech(access_token, txt, speaker_id, audio_format='wav'):
    """Returns audio URL for given text and speaker."""
    bit_rate = 128
    sample_rate = 22050
    url = f"{root_url}/speech"
    response = requests.request("GET", url,
                                params={'speaker_id': speaker_id,
                                        'txt': txt,
                                        'quality': 'high',
                                        'extension': audio_format,
                                        'bit_rate': bit_rate,
                                        'sample_rate': sample_rate},
                                headers={'Authorization': f'Bearer {access_token}'})
    response_text = response.text.encode('utf8')
    # print(response_text)
    response_json = json.loads(response_text)
    if response_json.get('error', None):
        print(f"\tERROR: {response_json['error']}, code: {response_json['error_code']}")
        raise Exception(f'Failed to generate "{txt}" with speaker {speaker_id}, error: {response_json["error_code"]}')
    if response_json.get('warning', None):
        print(f"\tWARNING: {response_json['warning']}")
    audio_url = response_json.get('url', None)
    return audio_url

def rivaserver(txt):
        access_token = get_access_token(client_id, secret)
        print(access_token)
        access_token_payload = decode_jwt(access_token)
        exp = access_token_payload.get("exp")  # expiry in Epoch time
        permissions = access_token_payload.get("scopes")
        print(f'Access token expiry: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp))}')
        print(f'Permissions: {permissions}')
        if 'generate' not in permissions:
            raise Exception('Missing "generate" permission. Did you run out of credits?')

        voices = get_voices(access_token)
        print(voices)
        voice = next((voice for voice in voices if voice['name'] == 'Thomas - Default'), voices[0])
        #voice = next((voice for voice in voices if voice['name'] == 'Amber - Neutral'), voices[0])
        print(f'Using voice: {voice["name"]} {voice["uuid"]}')

        speaker_id = voice["uuid"]

        audio_url = text_to_speech(access_token, txt, speaker_id, audio_format)

        print(f'Audio URL: {audio_url}')

        r = requests.get(audio_url)
        filename = voice['name'] + '.' + audio_format
        with open(filename, 'wb') as f:
            f.write(r.content)
        print(f'Saved to file: {filename}')


