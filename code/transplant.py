from utility import *
from mem_unit import *


def combine_mem(filepath):
    mem_list = []
    filepath = filepath / "encoders"
    for path in [x for x in filepath.iterdir() if x.is_dir()]:
        print(path)
        encoder = load_autoencoder(path)
        layers = encoder.layers
        mem = layers[(len(layers) // 2)].memory.weight
        mem_list.append(mem.numpy())

    final_mem = np.concatenate(mem_list)
    return final_mem


def create_transplanted_auto(
    layer_list, input_dims, final_mem, learning_rate=0.0001, shrink_thresh=0
):
    mem_module = MemModule(
        final_mem.shape[0], final_mem.shape[1], shrink_thresh=shrink_thresh
    )
    mem_unit = MemoryUnit(
        final_mem.shape[0], final_mem.shape[1], shrink_thresh=shrink_thresh
    )
    mem_unit.trainable = False
    mem_unit.set_weights([final_mem])

    mem_module.memory = mem_unit

    final_encoder = make_auto_encoder(
        layer_list, input_dims, learning_rate=learning_rate, mem_module=mem_module
    )
    return final_encoder


# @profile
def train_comp_2017(
    result_folder,
    auto_layers,
    dnn_layers,
    split=0.8,
    enc_epochs=150,
    enc_batch_size=512,
    enc_learning_rate=0.0001,
    dnn_epochs=150,
    dnn_batch_size=512,
    dnn_learning_rate=0.0001,
    mem_dim=1500,
    easy=True,
    attack=True,
    num_classes=15,
    max_num=None,
    shrink_thresh=0,
    redo=True,
    orig_filename=None,
):

    categories = [
        "BENIGN",
        "DoS Hulk",
        "PortScan",
        "DDoS",
        "DoS GoldenEye",
        "FTP-Patator",
        "SSH-Patator",
        "DoS slowloris",
        "DoS Slowhttptest",
        "Bot",
        "Web Attack Brute Force",
        "Web Attack XSS",
        "Infiltration",
        "Web Attack Sql Injection",
        "Heartbleed",
    ]

    filenames = get_CICIDS_2017()
    input_dims = 78

    result_folder = result_folder / "transplanted_mem"
    if result_folder.exists():
        if not redo:
            return
    else:
        mkdir(result_folder)

    if max_num:
        print("Max num is ", max_num)
        x_train, y_train, x_val, y_val, x_test, y_test = ret_cic2017(filenames)
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        rus = RandomUnderSampler({"BENIGN": max_num})
        x_train, y_train = rus.fit_resample(x_train, y_train)

        pass

    else:
        print("Got inside train cic2017 else")

        x_train, y_train, x_val, y_val, x_test, y_test = ret_cic2017(filenames)
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

    x_attack_val = None
    y_attack_val = None

    if attack and max_num:
        x_attack, y_attack = ret_specific_data(x_train, y_train, "BENIGN", True)
        x_attack_val, y_attack_val = ret_specific_data(x_val, y_val, "BENIGN", True)

        print("Got inside attack")

    else:

        x_attack, y_attack = x_train, y_train
        x_attack_val, y_attack_val = x_val, y_val
        print("Got inside all data")

    y_train, y_test, le = encode_labels(y_train, y_test)
    y_val = le.transform(y_val)
    labels = le.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    final_mem = combine_mem(orig_filename)
    dnn = make_dnn(dnn_layers, num_classes, dnn_learning_rate)

    auto_encoder = create_transplanted_auto(
        auto_layers, input_dims, final_mem, enc_learning_rate, shrink_thresh
    )

    # Train both models
    dnn, dnn_history, encoder, enc_history = val_split_training(
        x_train,
        x_attack,
        y_train,
        x_val,
        y_val,
        x_attack_val,
        auto_layers,
        auto_encoder,
        dnn,
        split,
        enc_epochs,
        enc_batch_size,
        dnn_epochs,
        dnn_batch_size,
    )

    x_test = pd.read_hdf(filenames.get("x_test"))
    class_string, class_dict, test_results = eval_model(
        encoder, dnn, x_test, y_test, labels, dnn_batch_size
    )

    # save_model(encoder, dnn, result_folder)
    save_model(auto_encoder, dnn, result_folder)

    # Write down results
    write_results(
        enc_history, dnn_history, test_results, class_string, class_dict, result_folder
    )


def COMP_CICIDS2017(line):
    full_args = line.split(",")
    args = full_args[1:]

    orig_filename = args[0]
    args[0] = result_folder_date(args[0])

    Path.mkdir(args[0], parents=True)
    enc_layers = list((int(x) for x in args[1].split(":")))
    dnn_layers = list(int(x) for x in args[2].split(":"))
    if args[14] == "None" or args[14] == "None\n":
        args[14] = None
    else:
        args[14] = int(args[14])

    train_comp_2017(
        args[0],
        enc_layers,
        dnn_layers,
        float(args[3]),
        int(args[4]),
        int(args[5]),
        float(args[6]),
        int(args[7]),
        int(args[8]),
        float(args[9]),
        int(args[10]),
        args[11],
        args[12],
        int(args[13]),
        args[14],
        float(args[15]),
        orig_filename=Path(orig_filename),
    )


def command_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("7"):
                COMP_CICIDS2017(line)


def main():
    if len(argv) >= 2:
        cleaned_path = argv[1].strip("'").strip('"')
        filename = Path(cleaned_path)
        print("This is filename ", filename)
        command_file(filename)
        train_CIC2017(result_folder, [110, 90], [60, 45], enc_epochs=1, dnn_epochs=1)
        print("Finished")

    else:
        print("Give input and output")
    pass


if __name__ == "__main__":
    main()
