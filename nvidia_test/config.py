from pydantic import BaseModel

ADJEN_SERVER_KEY="AQEqhmfxJ4/MaRxLw0m/n3Q5qf3VeIdIA4NGWXy9pun8GTQGkC63ub4ERMbPEMFdWw2+5HzctViMSCJMYAc=-0XaQtQXdd/CMrrVoJA4sukMVjRwxF7564IIixd2mCTU=-i1iD)AgpT:8b7[zYw#5"


OPENAI_API_KEY='sk-svcacct-wH6v9JbdE24mThslHk_iJhof6Y2Oh-DZin_L_2y6aMuHsixD2SZT3BlbkFJm-pjiJQ9602rUB1whbeQQTT_YgHkhVPDMbkC8v9pjgCuaPyKEQwA'







emotion_enum=["Happy","Sad","Lonely", "Indifferent","N/A"]
reply_enum=[ "Console", "Encourage", "Answer","Narrate", "N/A"]

response_schema={
  "name": "kids_emotion_reply",
  "strict": True,
  "schema": {
    "type": "object",
    "properties": {
      "emotion": {
        "type": "string",
        "description": "The emotion of the child.",
        "enum": emotion_enum
      },
      "reply_type": {
        "type": "string",
        "description": "The type of reply generated.",
        "enum": reply_enum
      },
      "text_reply": {
        "type": "string",
        "description": "The actual text of the reply."
      }
    },
    "required": [
      "emotion",
      "reply_type",
      "text_reply"
    ],
    "additionalProperties": False
  }
}

system_prompt_json="You are a nice, friendly and empathetic nanny, who takes care of kids emotion and replies with simple and kids-understandable words and sentences. "
"Kids might ask questions, talks nonsense, or whatever."            
"\n\nAnd you will"
"\nparse kids emotion (one of 5 possible types Happy, Sad, Lonely, Indifferent, or N/A)"
"\ngive reply type (one of 5 possible types Console, Encourage, Answer, Narrate, or N/A)"            
"\nand give actual text reply."
"\n\nYou response should be a JSON with 3 entries emotion, reply_type, and text_reply. formatted like below:"          
"\n```\n{\"emotion\":\"...\",\"reply_type\":\"...\",\"text_reply\":\"...\"}\n```"

system_prompt="You are a nice, friendly and empathetic nanny, who takes care of kids emotion and replies with simple and kids-understandable words and sentences. "
"Kids might ask questions, talks nonsense, or whatever."            
"\n\nAnd you must just give the actual text response, without any additional explanatory text."