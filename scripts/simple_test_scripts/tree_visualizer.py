import pygame

WIDTH = 1800
HEIGHT = 600

pygame.init()
pygame.display.set_caption('Tree ╰*°▽°*╯')
window = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

running = True

mouse_down = False
space_down = False


class Node:
    
    def __init__(self, identity, node_type):
        self.id = identity
        self.type = node_type


def read_tree(file_name):
    
    nodes = []
    
    with open(file_name, 'r') as file:
        data = file.read()
        node_strings = data.split('{')
    
    for node in node_strings[1:]:
        params = node.split(',')
        identity = int(params[0][11:])
        
        if params[1][0:7] == ' "leaf"':
            node_type = 'leaf'
            value = float(params[1][9:15])
            new_node = Node(identity, node_type)
            new_node.value = value
            nodes.append(new_node)
        
        else:
            depth = int(params[1][9:])
            node_type = 'split'
            split = params[2][11:-1]
            yes = params[4][7:]
            no = params[5][6:]
            
            new_node = Node(identity, node_type)
            new_node.split = split
            new_node.yes = int(yes)
            new_node.no = int(no)
            new_node.depth = depth
            nodes.append(new_node)
    
    return nodes

nodes = read_tree('tree_data.json')

def get_node_pos(node):
    depth_max = 2**node.depth
    width_i = node.id-depth_max+1
    return (WIDTH/2 - ((depth_max/2 - width_i)*75)), node.depth*100


while(running):
    
    window.fill((0, 0, 0))
    
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
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
                space_down = False
    
    
    for i, node in enumerate(nodes):
        if node.type == 'leaf':
            pygame.draw.circle(window, (110, 50, 170), (100, 100), 10)
        else:
            depth_max = 2**node.depth
            width_i = node.id-depth_max+1
            pygame.draw.rect(window, (200, 50+node.depth*5, 50+node.id*3), pygame.Rect(get_node_pos(node), (50, 50)))
            
            yes_node = nodes[node.yes]
            if yes_node.type != 'leaf':
                pygame.draw.line(window, (0, 170, 0), get_node_pos(node), get_node_pos(yes_node))
    
    pygame.display.flip()

pygame.quit()
