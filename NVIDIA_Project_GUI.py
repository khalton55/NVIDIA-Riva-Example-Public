# Imports
import gradio as gr
import base64
##########################
import io
import numpy as np
import riva.client

config = {
    'English': {
        'langCode': 'en-US',
        'megatronLang': 'en',
        'ttsVoiceString': {
            'male': {
                'normal': 'English-US.Male-1',
                'calm': 'English-US.Male-Calm',
                'neutral': 'English-US.Male-Neutral',
                'happy': 'English-US.Male-Happy',
                'angry': 'English-US.Male-Angry'
            },
            'female': {
                'normal': 'English-US.Female-1',
                'calm': 'English-US.Female-Calm',
                'neutral': 'English-US.Female-Neutral',
                'happy': 'English-US.Female-Happy',
                'angry': 'English-US.Female-Angry',
                'fearful': 'English-US.Female-Fearful',
                'sad': 'English-US.Female-Sad'
            }
        }
          },
    'Spanish': {
        'langCode': 'es-US',
        'megatronLang': 'es',
        'ttsVoiceString': {
            'male': {
                'normal': 'Spanish-US.Male-1',
                'calm': 'Spanish-US.Male-Calm',
                'neutral': 'Spanish-US.Male-Neutral',
                'happy': 'Spanish-US.Male-Happy',
                'angry': 'Spanish-US.Male-Angry',
                'fearful': 'Spanish-US.Male-Fearful' ,
                'sad': 'Spanish-US.Male-Sad'
            },
            'female': {
                'normal': 'Spanish-US.Female-1',
                'calm': 'Spanish-US.Female-Calm',
                'neutral': 'Spanish-US.Female-Neutral',
                'angry': 'Spanish-US.Female-Angry',
                'sad': 'Spanish-US.Female-Sad'
            }
        }
          },
    # TODO: Add sentiment and gender for remaining languages
    "German": {
        'langCode': 'de-DE',
        'megatronLang': 'de',
        'ttsVoiceString': {
            'male': {
                'normal': 'German-DE-Male-1'
            }
        }
          },
    "Italian": {
        'langCode': 'it-IT',
        'megatronLang': 'it',
        'ttsVoiceString': {
            'male': {
                'normal': 'Italian-IT-Male-1'
            }
        }
          },
    'Mandarin': {
        'langCode': 'zh-CN',
        'megatronLang': 'zh',
        'ttsVoiceString': {
            'male': {
                'normal': 'Mandarin-CN.Male-1'
            }
        }
          },
}

def translation(inputLanguage, outputLanguage, outputGender, outputSentiment, audio):
   
    warning = False
    if inputLanguage == None:
         gr.Warning('No input language selected!')
         warning = True
    if outputLanguage == None:
        gr.Warning('No output language selected!')
        warning = True

    if outputGender == None:
        gr.Warning('No output gender selected!')
        warning = True
    if outputSentiment == None:
        gr.Warning('No output sentiment selected!')
        warning = True
    if audio == None:
        gr.Warning('No audio provided!')
        warning = True

    if inputLanguage == outputLanguage:
        gr.Warning('Input and Output Languages must be different!')
        warning = True

    if warning == True:
        return "Check Warnings!"
    
    #Authentication

    #ENTER RIVA SERVER IP
    #CURRENTLY SUPPORTS ENGLISH <--> SPANISH
    #NEEDS EN + ES-US ASR and TTS
    #ALSO NEEDS 'megatronnmt_any_any_1b' Megatron NMT Model (Any to any)
    authRiva = riva.client.Auth(uri='ENTER URL')
    


    #Service Setup
    riva_asr = riva.client.ASRService(authRiva)
    riva_tts = riva.client.SpeechSynthesisService(authRiva)
    riva_nmt_client = riva.client.NeuralMachineTranslationClient(authRiva)

    path = audio
    with io.open(path, 'rb') as fh:
        content = fh.read()

    langCodeASR = config[inputLanguage]['langCode']

    # Set up an offline/batch recognition request
    asrConfig = riva.client.RecognitionConfig()

    # Language code of the audio clip
    asrConfig.language_code = langCodeASR                  

    # How many top-N hypotheses to return
    asrConfig.max_alternatives = 1

    # Add punctuation when end of VAD detected                       
    asrConfig.enable_automatic_punctuation = True

    # Mono channel       
    asrConfig.audio_channel_count = 1                    

    #Creates List of sentences
    response = riva_asr.offline_recognize(content, asrConfig)
    asr_best_transcripts = []
    i=0
    while i < len(response.results):
        asr_best_transcripts.append(response.results[i].alternatives[0].transcript)
        i= i + 1


    model_name = 'megatronnmt_any_any_1b'

    
    source_language = config[inputLanguage]['megatronLang']
    target_language = config[outputLanguage]['megatronLang']

    response = riva_nmt_client.translate(asr_best_transcripts, model_name, source_language, target_language)

    ttsLangCode = config[outputLanguage]['langCode']
    ttsVoice = config.get(outputLanguage, []).get('ttsVoiceString', []).get(outputGender.lower(), []).get(outputSentiment.lower(), [])
    sample_rate_hz = 44100
    req = { 
            "language_code"  : ttsLangCode,
            "encoding"       : riva.client.AudioEncoding.LINEAR_PCM ,   # LINEAR_PCM and OGGOPUS encodings are supported
            "sample_rate_hz" : sample_rate_hz,                          # Generate 44.1KHz audio
            "voice_name"     : ttsVoice                   # The name of the voice to generate
    }

    i = 0
    audio_samples = ''
    full_audio = ''

    # TTS on each of the translations and combines audio
    while i < len(response.translations):
        req["text"] = response.translations[i].text
        resp = riva_tts.synthesize(**req)
        audio_samples = np.frombuffer(resp.audio, dtype=np.int16)
        
        if i==0:
            full_audio = audio_samples
        else:
            full_audio = np.hstack((full_audio, audio_samples))
        i = i + 1
    
    return (sample_rate_hz, full_audio)

def updateGender(language):
    genderList = list(config.get(language, []).get('ttsVoiceString', []).keys())
    capGenderList = [s.capitalize() for s in genderList]
    
    return [gr.Dropdown(choices=capGenderList, interactive=True, value=None), gr.Dropdown(choices=[], interactive=True, value=None)]

def updateSentiment(language, gender):
    sentimentList = list(config.get(language, []).get('ttsVoiceString', []).get(gender.lower(), []).keys())
    capSentimentList = [s.capitalize() for s in sentimentList]

    return gr.Dropdown(choices=capSentimentList, interactive=True, value=None)

#Annoying file access issue for directly referencing local file (Probably a better way but this works)
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

    
with gr.Blocks() as app:
    image_path = './LinguaPic_Logo.webp'
    img_base64 = get_image_base64(image_path)
    gr.HTML(f'''
            <h1 style='text-align: center;'>LinguaPic </h1>
            <p style="text-align: center;">Input a picture of yourself and and audio file of you speaking to create a video translation into one of many languages. This service utilizes NVIDIA\'s APIs to function.</p>
            <img src="data:image/jpeg;base64,{img_base64}" width=\"500\" height=\"500\" style=\"display: block; margin-left: auto; margin-right: auto;\" >
            ''')
    
    # Updates list to edit which languages are supported depending on RIVA server
    languages = ["English","Spanish", "German" ]
    dropDownInput = gr.Dropdown(
        choices=languages, label="Input Language", info="Pick language spoken in audio file"
        )
    dropDownOuput = gr.Dropdown(
        choices=languages, label="Output Language", info="Pick language to be spoken in the output audio"
        )
    
    dropDownOuputGender = gr.Dropdown(
        choices=[], label='Output Gender', info="Pick Gender of TTS "
    )

    dropDownOuputSentiment = gr.Dropdown(
        choices=[], label='Output Sentiment', info='Pick Sentiment of TTS'
    )

    dropDownOuput.input(
        lambda language: updateGender(language),
        inputs=[dropDownOuput],
        outputs=[dropDownOuputGender, dropDownOuputSentiment]
    )

    dropDownOuputGender.input(
        lambda language, gender: updateSentiment(language, gender) ,
        inputs=[dropDownOuput, dropDownOuputGender],
        outputs=[dropDownOuputSentiment]
    )

    audio = gr.Audio(format='wav', type='filepath', label='Audio File to Translate')
    output = gr.Audio(label="Translation Output")
    submitButton = gr.Button('Submit')

    submitButton.click(
        fn=translation,
        inputs=[dropDownInput, dropDownOuput, dropDownOuputGender, dropDownOuputSentiment, audio],
        outputs=[output]
    )

app.launch()