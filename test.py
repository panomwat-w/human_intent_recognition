def calc_brick_origin(max_num_bricks=5, brick_origin=(0.0,0.0), brick_dim=(0.06,0.12), margin=0.006):
    fix_brick_origin_list = []
    for i in range(1, max_num_bricks + 1):
        if i % 2 != 0:
            fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(i-1), brick_origin[1]))
            for j in range(1, int((i-1)/2) + 1):
                fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(i-1), brick_origin[1]+(brick_dim[1]+margin)*j))
                fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(i-1), brick_origin[1]-(brick_dim[1]+margin)*j))
            if i < max_num_bricks:
                fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(2 * max_num_bricks - i - 1), brick_origin[1]))
                for j in range(1, int((i-1)/2) + 1):
                    fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(2 * max_num_bricks - i - 1), brick_origin[1]+(brick_dim[1]+margin)*j))
                    fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(2 * max_num_bricks - i - 1), brick_origin[1]-(brick_dim[1]+margin)*j))
        else:
            for j in range(1, int(i/2) + 1):
                fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(i-1), brick_origin[1]+(brick_dim[1]+margin)*(j-0.5)))
                fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(i-1), brick_origin[1]-(brick_dim[1]+margin)*(j-0.5)))
                if i < max_num_bricks:
                    fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(2 * max_num_bricks - i - 1), brick_origin[1]+(brick_dim[1]+margin)*(j-0.5)))
                    fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(2 * max_num_bricks - i - 1), brick_origin[1]-(brick_dim[1]+margin)*(j-0.5)))
    return fix_brick_origin_list

if __name__ == '__main__':
    n = int(input("Enter the max number of bricks: "))
    print(calc_brick_origin(n, (0.0,0.0), (1,2), 0.0))