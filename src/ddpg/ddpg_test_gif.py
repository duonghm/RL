import gym
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFont
import PIL.ImageDraw as ImageDraw
import os
import imageio
import glob


def policy(env, act, state):
    upper_bond = env.action_space.high[0]
    lower_bond = env.action_space.low[0]
    act_sample = np.squeeze(act(state))
    act_sample = np.clip(act_sample, lower_bond, upper_bond)
    return [act_sample]


font = ImageFont.truetype("arial.ttf", 32)


def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    if np.mean(im) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)

    drawer.text((im.size[0] / 20, im.size[1] / 18), f'Episode: {episode_num}', font=font, fill=text_color)

    return im


def save_frames_as_mp4(frames, output_location, fps=24):
    imageio.mimwrite(output_location, frames, fps=fps)


if __name__ == '__main__':
    frames = []
    print(glob.glob('ddpg_models/20210913_170332/*'))
    for mdl_folder in glob.glob('ddpg_models/20210913_170332/*'):
        ep_id = mdl_folder.split('\\')[-1]
        print(mdl_folder, ep_id)
        actor = tf.keras.models.load_model(f'{mdl_folder}/actor')
        actor.summary()
        env = gym.make('Pendulum-v0')
        state = env.reset()
        while True:
            frame = env.render(mode='rgb_array')
            frames.append(_label_with_episode_number(frame, ep_id))
            state = tf.expand_dims(state, 0)
            action = policy(env, actor, state)
            # action = np.random.uniform(-0.01, 0.01, 1)
            state, reward, done, info = env.step(action)
            if done:
                break
        env.close()
    # _save_frames_as_gif(frames, path='./videos/', filename='ep49.gif')
    print('write video ...')
    save_frames_as_mp4(frames, 'videos/pendulum.mp4', 30)
