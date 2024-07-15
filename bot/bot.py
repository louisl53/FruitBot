# Importing necessary libraries and functions for the bot's functionality
from agent_gpt import gpt_intention, gpt_yes_no, gpt_question, gpt_recipe, chat_gpt
import requests
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackContext, CallbackQueryHandler, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv
import os

# Loading environment variables from .env file
load_dotenv()
token = os.getenv("TOKEN")  # Bot's unique token from BotFather

# Setting up default values for API host and port, used to communicate with the backend service
api_host = os.getenv('API_HOST', '127.0.0.1')
api_port = os.getenv('API_PORT', '8080')
url = f"http://{api_host}:{api_port}/predict/"  # Full URL for making POST requests

# Function to start conversation, sends an introductory message when the user starts interaction
async def start(update: Update, context: CallbackContext) -> None:
    intro_message = """Hello! I am FruitsBot, your assistant to help you determine if your fruits and vegetables are fresh.
    Send me an image of your fruits and I will tell you if they are good. I can also suggest suitable recipes! 😇"""
    await update.message.reply_text(intro_message)

# Function to handle the receipt of a photo from the user
async def handle_photo(update: Update, context: CallbackContext) -> None:
    photo = update.message.photo[-1]  # Accessing the last photo sent by the user
    file_id = photo.file_id  # Unique identifier for the photo file
    file = await context.bot.get_file(file_id)  # Getting the file from Telegram
    file_path = f"Image_message/{file_id}.jpg"  # Defining file path for saving
    await file.download_to_drive(file_path)  # Downloading the file to the local drive
    
    # Opening the image and sending it to the prediction API
    with open(file_path, 'rb') as image:
        response = requests.post(url, files={'file': image})
    
    # Checking response status and processing accordingly
    if response.status_code == 200:
        predicted_class = response.json()['predicted_class']
        context.user_data['predicted_class'] = predicted_class
        gpt = gpt_yes_no(predicted_class)  # Generating a response based on the prediction
        keyboard = [
            [InlineKeyboardButton("Yes", callback_data='yes'),
             InlineKeyboardButton("No", callback_data='no')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)  # Creating inline buttons for user interaction
        await update.message.reply_text(gpt, reply_markup=reply_markup)
    else:
        await update.message.reply_text('Sorry, I couldn\'t process the image.')  # Error handling if image can't be processed

    os.remove(file_path)  # Cleaning up by removing the downloaded file



# Handling responses to callback queries from inline buttons
async def handle_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()  # Notifying Telegram that the callback query was received

    # Retrieve 'predicted_class' from user data, checking if it's stored
    predicted_class = context.user_data.get('predicted_class', None)
    if predicted_class is None:
        # Informing the user about the missing data in case of an error
        await query.edit_message_text("I'm sorry, there seems to be an error. I can't find the necessary information.")
        return

    # Respond based on the user's selection in the inline keyboard
    if query.data == 'yes':
        recipe = gpt_question(predicted_class)  # Generate a recipe suggestion using the predicted class
        await query.edit_message_text(recipe)
    elif query.data == 'no':
        await query.edit_message_text("Okay, if you have any other fruits or vegetables you'd like to show me, don't hesitate to do so. 😉")
        # Clean up user data to prevent stale data usage
        if 'predicted_class' in context.user_data:
            del context.user_data['predicted_class']

# Function to handle text messages sent by the user
async def handle_text_message(update: Update, context: CallbackContext) -> None:
    # Retrieve or set a default value for 'predicted_class'
    predicted_class = context.user_data.get('predicted_class', 'no_class')

    text = update.message.text  # Capturing the user's text message

    # Determine the intention behind the user's message using GPT
    intent = gpt_intention(text)

    # Process the message based on the identified intent
    if intent == "recipe":
        new_recipe = gpt_recipe(text, predicted_class)  # Generate a recipe based on the text and predicted class
        await update.message.reply_text(new_recipe)
    elif intent == "chat":
        gpt_answer = chat_gpt(text)  # Generate a conversational response using GPT
        await update.message.reply_text(gpt_answer)

    # Clean up user data after processing
    if 'predicted_class' in context.user_data:
        del context.user_data['predicted_class']

# Main function to initialize the bot and handle commands and messages
def main():
    TOKEN = token  # Bot token obtained from BotFather
    
    # Set up the Telegram bot application
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT, handle_text_message))
    application.add_handler(CallbackQueryHandler(handle_callback))

    application.run_polling()  # Start polling for messages

# Entry point of the script
if __name__ == '__main__':
    # Ensure the directory for storing images exists
    if not os.path.exists("Image_message"):
        os.makedirs('Image_message')
    main()  # Run the main function

