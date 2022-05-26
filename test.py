def generate_video_response(text_input):
    # Let's chat for 6 lines
    global chat_history_ids
    global step

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input("User:") + tokenizer.eos_token, return_tensors='pt')
    # print(new_user_input_ids)# append the new user input tokens to the chat history
    if step % 5 == 0:
        step = 0
        chat_history_ids = None

    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    chat_history_ids = model_text.generate(
        bot_input_ids, max_length=1024,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=4,
        do_sample=True,
        top_k=200,
        top_p=0.5,
        temperature=0.2
    )
    answer = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0])
    for letter in answer[0:4]:
        if letter in ['!', '?', '.', ',,']:
            bot_input_ids = None
            chat_history_ids = None
            step = 0
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids],
                                      dim=-1) if step > 0 else new_user_input_ids
            chat_history_ids = model_text.generate(
                bot_input_ids, max_length=1024,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=4,
                do_sample=True,
                top_k=200,
                top_p=0.5,
                temperature=0.2
            )
            answer = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0])

    # pretty print last ouput tokens from bot
    step += 1
    prompt = "{}".format(answer, skip_special_tokens=True).strip("<|endoftext|>")
    print(prompt)
    os.system(f'tts --model_name tts_models/en/ljspeech/tacotron2-DDC --text "{prompt}" --out_path audio.wav')
    driving_spec = compute_spec('audio.wav')
    y_lengths = torch.tensor([driving_spec.size(-1)])
    ref_wav_voc, _, _ = model.voice_conversion(driving_spec, y_lengths, driving_emb, target_emb)
    ref_wav_voc = ref_wav_voc.squeeze().detach().numpy()
    with open('/content/MakeItTalk/examples/audio_pim.wav', 'wb') as f:
        f.write(Audio(ref_wav_voc, rate=ap.sample_rate).data)
    os.chdir('MakeItTalk')
    os.system('sudo python main_end2end_cartoon.py --jpg pim.png --jpg_bg pim_bg.jpg --inner_lip')
    os.chdir('..')
