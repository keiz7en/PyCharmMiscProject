import requests

BOT_TOKEN = '8162909553:AAG0w8qjXLQ3Vs2khbTwWq5DkRcJ0P29ZkY'


def get_chat_id():
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    response = requests.get(url)
    data = response.json()

    print("Bot Response:")
    print("=" * 50)

    if data.get('ok') and data.get('result'):
        for update in data['result']:
            if 'message' in update:
                chat = update['message']['chat']
                print(f"Chat ID: {chat['id']}")
                print(f"Chat Type: {chat['type']}")
                if 'first_name' in chat:
                    print(f"Name: {chat['first_name']}")
                if 'username' in chat:
                    print(f"Username: @{chat['username']}")
                print("-" * 30)
    else:
        print("No messages found. Please:")
        print("1. Start a chat with @mfion_bot")
        print("2. Send any message (like /start)")
        print("3. Run this script again")

    print("\nFull response:")
    print(data)


if __name__ == "__main__":
    get_chat_id()
