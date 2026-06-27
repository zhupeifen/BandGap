import pygame
import numpy as np
import xgboost as xgb
from scipy.ndimage import shift

WIDTH = 800
HEIGHT = 600

pygame.init()
pygame.display.set_caption('Picture Identifier')
window = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

running = True

mouse_down = False
space_down = False
erase = False

res = (16, 16)
pixels = np.zeros(res)

directory = 'smiles'

SHAPE_NAMES = ['circle', 'cross', 'smile :)']


def save_drawing():
    global pixels
    
    with open(directory + '/data.txt', 'a') as file:
        line = ''
        for x in range(16):
            for y in range(16):
                line += str(int(pixels[x, y]))
        
        print(line, file=file)
    
    pixels = np.zeros(res)
    print('saved')


objectives = ['binary:logistic', 'multi:softprob']
MULTIPLE_SHAPES = True


model = xgb.XGBClassifier(
    n_estimators=55,
    max_depth=3,
    learning_rate=0.02,
    subsample=0.8,
    objective=objectives[int(MULTIPLE_SHAPES)],
    random_state=42
)

def load_training_data():
    global model, pixels
    
    total_shapes = 0
    num_circles = sum(1 for line in open('circles/data.txt', 'r') if line.strip())
    num_crosses = sum(1 for line in open('crosses/data.txt', 'r') if line.strip())
    num_smiles = sum(1 for line in open('smiles/data.txt', 'r') if line.strip())
    
    total_shapes = num_circles+num_crosses
    if MULTIPLE_SHAPES:
        total_shapes += num_smiles
    
    print('circle count: ' + str(num_circles))
    print('cross count: ' + str(num_crosses))
    print('smile count: ' + str(num_smiles))
    
    total_shapes *= 21
    train_X = np.zeros((total_shapes, res[0]*res[1]))
    train_y = np.zeros((total_shapes))
    
    index = 0
    files = ['circles/data.txt', 'crosses/data.txt']
    if MULTIPLE_SHAPES:
        files.append('smiles/data.txt')
    
    for shape_num, file_name in enumerate(files):
        with open(file_name, 'r') as file:
            for line in file:
                all_zero = True
                arr = np.zeros(res[0]*res[1])
                for i, num in enumerate(line):
                    if num == '0' or num == '1':
                        arr[i] = int(num)
                        if num == '1':
                            all_zero = False
                
                if all_zero:
                    print(file_name + ' has a zero line')
                
                train_X[index] = arr
                train_y[index] = 0
                index += 1
                
                arr = arr.reshape(16, 16)
                for x in range(-2, 3):
                    for y in [-2, -1, 1, 2]:
                        train_X[index] = shift(arr, shift=[x, y], cval=0).flatten()
                        train_y[index] = shape_num
                        index += 1
    
    model.fit(train_X, train_y)
    model.save_model('image_detection_model.json')


def load_training_data_from_file():
    global model
    model.load_model('image_detection_model.json')


MODE = 'guessing'
load_training_data_from_file()


def guess_drawing():
    global pixels
    
    arr = np.zeros((1, res[0]*res[1]))
    
    for x in range(res[0]*res[1]):
        arr[0, x] = pixels[x//16, x%16]
    
    prediction = model.predict(arr)
    print(SHAPE_NAMES[prediction[0]])
    print(model.predict_proba(arr)[0])
    
    pixels = np.zeros((res))


def update_drawing():
    mouse_pos = pygame.mouse.get_pos()
    
    sw = WIDTH/res[0]
    sh = HEIGHT/res[1]
    
    if mouse_pos[0] < WIDTH and mouse_pos[1] < HEIGHT and mouse_down:
        pixels[int(mouse_pos[0]/sw), int(mouse_pos[1]/sh)] = not erase;


while(running):
    
    window.fill((0, 0, 0))
    
    sw = WIDTH/res[0]
    sh = HEIGHT/res[1]
    
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if pixels[int(mouse_pos[0]/sw), int(mouse_pos[1]/sh)] == 1:
                pixels[int(mouse_pos[0]/sw), int(mouse_pos[1]/sh)] = 0
                erase = True
            else:
                pixels[int(mouse_pos[0]/sw), int(mouse_pos[1]/sh)] = 1
                erase = False
            mouse_down = True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            
            elif event.key == pygame.K_SPACE:
                space_down = True
        
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                if space_down:
                    if MODE == 'saving':
                        save_drawing()
                    elif MODE == 'guessing':
                        guess_drawing()
                
                space_down = False
    
    update_drawing()
    
    for x in range(res[0]):
        for y in range(res[1]):
            if pixels[x, y] == 1:
                pygame.draw.rect(window, (200, 200, 200), pygame.Rect(x*sw, y*sh, sw, sh))
    
    pygame.display.flip()

pygame.quit()