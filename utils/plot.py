from matplotlib import pyplot as plt


def plot_losses(train_losses,val_losses,val_losses_gen, val_losses_ext_grid, case_name, title, save_path):
    plt.close()
    plt.cla()
    plt.clf()

    fig = plt.figure()
    plt.figure().clear()

    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="validation loss")
    plt.plot(val_losses_gen, label="validation loss generators")
    plt.plot(val_losses_ext_grid, label="validation loss ext_grid")

    plt.xlabel("training epoch")
    plt.ylabel("Error")
    plt.title('p_mw and q_mvar on '+case_name+" "+title)
    plt.legend()
    plt.savefig(save_path+"/losses.png")
    print("over")