import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
import time

torch.manual_seed(42)
weight = models.ResNet18_Weights.IMAGENET1K_V1
deep_weight = models.ResNet50_Weights.IMAGENET1K_V1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def target_to_binary(label:int) -> int:
    if label in [1,6,7,8,10,12,21,24,27,28,33,34]:
        return 0 # cat
    return 1 #dog

def split_data(dataset):
    
    n = len(dataset)
    n_classes = 37
    train_indices = []
    val_indices = []


    class_dist = {}
    for i in range(n_classes):
        class_dist[i] = [0,0,0] #train, val, total
    for i, data in enumerate(dataset):
        breed = data[1]
        animal_type = target_to_binary(breed) 

        #we want 20% of all cats in train and 80% of all dogs in train
        if animal_type == 0: #cat
            targets = [0.2, 0.2]
        else: #dog
            targets = [0.8,0.2]
        

        if class_dist[breed][2] == 0 or (class_dist[breed][0]/class_dist[breed][2]) < targets[0]: #add to train
            train_indices.append(i)
            class_dist[breed][0] += 1
        elif (class_dist[breed][1] / class_dist[breed][2]) < targets[1]: #add to val
            val_indices.append(i)
            class_dist[breed][1] += 1

        class_dist[breed][2] += 1

        
    train = torch.utils.data.Subset(dataset, train_indices)
    val = torch.utils.data.Subset(dataset, val_indices)

    print("dataset split with proportions train:", str(len(train)/n), "val: ", str(len(val)/n))
    target = 1/n_classes
    #weight * class_dist[i][i] = target
    weights = torch.tensor([(target/class_dist[i][0]) for i in list(range(37))])
    weights = torch.div(weights, sum(weights))
    print("Weights: ", weights)
    return train, val, weights


# Dataset loading
train_dataset_binary = datasets.OxfordIIITPet(root='./data', split='trainval', target_types='binary-category', download=True, transform=weight.transforms())
test_dataset_binary = datasets.OxfordIIITPet(root='./data', split='test', target_types='binary-category', download=True, transform=weight.transforms())
train_dataset_multi = datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=weight.transforms())
train_dataset_multi_transform = datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=train_transform)
train_dataset_multi_deep = datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=deep_weight.transforms())
test_dataset_multi = datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=weight.transforms())
test_dataset_multi = datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=weight.transforms())
test_dataset_multi_deep = datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=deep_weight.transforms())

train_imbalanced, val_imbalanced, imbalanced_weights = split_data(test_dataset_multi)

train_size = int(0.8 * len(train_dataset_binary))
val_size = len(train_dataset_binary) - train_size
train_dataset_binary, val_dataset_binary = torch.utils.data.random_split(train_dataset_binary, [train_size, val_size])
train_size = int(0.8 * len(train_dataset_multi))
val_size = len(train_dataset_multi) - train_size
train_dataset_multi, val_dataset_multi = torch.utils.data.random_split(train_dataset_multi, [train_size, val_size])
train_size = int(0.8 * len(train_dataset_multi_transform))
val_size = len(train_dataset_multi_transform) - train_size
train_dataset_multi_transform, val_dataset_multi_transform = torch.utils.data.random_split(train_dataset_multi_transform, [train_size, val_size])
train_size = int(0.8 * len(train_dataset_multi_deep))
val_size = len(train_dataset_multi_deep) - train_size
train_dataset_multi_deep, val_dataset_multi_deep = torch.utils.data.random_split(train_dataset_multi_deep, [train_size, val_size])

train_loader_binary = DataLoader(train_dataset_binary, batch_size=32, shuffle=True)
val_loader_binary = DataLoader(val_dataset_binary, batch_size=32, shuffle=False)
test_loader_binary = DataLoader(test_dataset_binary, batch_size=32, shuffle=False)
train_loader_multi = DataLoader(train_dataset_multi, batch_size=32, shuffle=True)
val_loader_multi = DataLoader(val_dataset_multi, batch_size=32, shuffle=False)
train_loader_multi_transform = DataLoader(train_dataset_multi_transform, batch_size=32, shuffle=True)
val_loader_multi_transform = DataLoader(val_dataset_multi_transform, batch_size=32, shuffle=False)
train_loader_multi_deep = DataLoader(train_dataset_multi_deep, batch_size=32, shuffle=True)
val_loader_multi_deep = DataLoader(val_dataset_multi_deep, batch_size=32, shuffle=False)
train_loader_imbalance = DataLoader(train_imbalanced, batch_size=32, shuffle=True)
val_loader_imbalance = DataLoader(val_imbalanced, batch_size=32, shuffle=False)
test_loader_multi = DataLoader(test_dataset_multi, batch_size=32, shuffle=False)
test_loader_multi_deep = DataLoader(test_dataset_multi_deep, batch_size=32, shuffle=False)

original_model = models.resnet18(weights=weight)
original_model = original_model.to(device)

criterion = nn.CrossEntropyLoss()
binary_criterion = nn.BCELoss()

train_losses = []
val_losses = []
accuracies = []

def calculate_accuracy(model, dataloader, criterion, binary):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if binary:
                outputs = outputs.squeeze()
                labels = labels.to(torch.float32)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                total += labels.size(0)
                #print("outputs: ", outputs)
                #print("labels: ", labels)
                correct += (torch.round(outputs) == labels).sum().item()
                #print("correct predictions:", (torch.round(outputs) == labels).sum().item())
            else: 
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = running_loss / len(dataloader)

    return accuracy, avg_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, binary, epochs=10):
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model.train()

    train_losses = []
    val_losses = []
    accuracies = []
    start_time = time.time()

    for epoch in range(epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if binary:
                outputs = torch.flatten(outputs)
                labels = labels.to(torch.float32)
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)
             
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)

        if not binary:
            scheduler.step()
        
        val_accuracy, val_loss = calculate_accuracy(model, val_loader, criterion, binary)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy * 100:.2f}%')
        print('-' * 20)
    
    training_time = time.time() - start_time
    print(f'Training time: {training_time}')

    return train_losses, val_losses, accuracies, training_time

def freeze_all_but_fc(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

def freeze_batchnorm_stats(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

def train_with_unfreezing(model, train_loader, val_loader, criterion, high_lr, reg, diff, epochs=10, unfreeze_interval = 2):
    freeze_all_but_fc(model)
    freeze_batchnorm_stats(model)

    layers_to_unfreeze = [model.layer4, model.layer3, model.layer2, model.layer1]

    if diff:
        layer_lrs = [1e-2, 1e-3, 1e-4, 1e-5]
    else:
        layer_lrs = [1e-3, 1e-3, 1e-3, 1e-3]

    if high_lr:
        layer_lrs = layer_lrs*10

    train_losses = []
    val_losses = []
    accuracies = []
    start_time = time.time()
    num_epochs = epochs

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0

        idx = epoch // unfreeze_interval
        if epoch % unfreeze_interval == 0 and idx < len(layers_to_unfreeze):
            new_layer = layers_to_unfreeze[idx]
            for param in new_layer.parameters():
                param.requires_grad = True
            print(f"Epoch {epoch+1}: Unfroze layer {idx + 1} ({new_layer._get_name()})")

        optimizer_params = [{'params': model.fc.parameters(), 'lr': 1e-3}]
        for i in range(min(idx + 1, len(layers_to_unfreeze))):
            optimizer_params.append({'params': layers_to_unfreeze[i].parameters(), 'lr': layer_lrs[i]})
        if reg:
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.01)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss/len(train_loader)

        scheduler.step()

        val_accuracy, val_loss = calculate_accuracy(model, val_loader, criterion, binary=False)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy * 100:.2f}%')
        print('-' * 20)
        

    training_time = time.time() - start_time
    print(f'Training time: {training_time}')

    return train_losses, val_losses, accuracies, training_time

def fine_tune_layers(model, num_layers_to_train):
    num_layers_to_train += 1
    
    for param in model.parameters():
        param.requires_grad = False
    for layer in list(model.children())[-num_layers_to_train:] + [model.fc]:
        for param in layer.parameters():
            # print(layer)
            param.requires_grad = True

def print_losses(training_costs, validation_costs, name = None):
    plt.clf()
    plt.plot(
        [i for i in range(len(training_costs))],
        training_costs,
        label=f"Training loss",
    )
    plt.plot(
        [i for i in range(len(validation_costs))],
        validation_costs,
        label=f"Validation loss",
    )
    plt.title("Loss of training and validation data")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Loss")
    plt.legend()
    if name is not None:
        plt.savefig("./plots/" + name + "_loss.png")
    else: 
        plt.show()

def print_acc(validation_accuracy, name = None):
    plt.clf()
    plt.plot(
        [i for i in range(len(validation_accuracy))],
        validation_accuracy,
        label="Validation accuracy",
    )
    plt.title("Accuracy of validation data")
    plt.xlabel("Evaluation step")
    plt.ylabel("Accuracy")
    plt.legend()
    if name is not None:
        plt.savefig("./plots/" + name + "_acc.png")
    else: 
        plt.show()

def binary():
    train_losses = []
    val_losses = []
    accuracies = []

    model = copy.deepcopy(original_model)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 1), nn.Sigmoid())
    model = model.to(device)
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    print(binary_criterion._get_name())
    fine_tune_layers(model, 0)
    print("Training for binary classification (Dog vs Cat)...")
    train_losses, val_losses, accuracies, train_time = train_model(model, train_loader_binary, val_loader_binary, binary_criterion, optimizer, binary=True, epochs=10)

    print("Calculating test accuracy for binary classification...")
    acc, _ = calculate_accuracy(model, test_loader_binary, binary_criterion, binary=True)
    print(f"Test accuracy: {acc * 100:.2f}%")

    print_losses(train_losses, val_losses, "binary")
    print_acc(accuracies, "binary")

    out_string = "final accuracy = " + str(acc) + ", training time = " + str(train_time) + " seconds"
    with open("binary.txt", "w") as f:
        f.write(out_string)

def strategy_one():
    layers = 5
    out_strings = []
    for l in range(1, layers + 1):
        train_losses = []
        val_losses = []
        accuracies = []

        model = copy.deepcopy(original_model)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_features, 37))
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)

        print('-' * 20)
        fine_tune_layers(model, l)
        print(f"Training for multi-class classification with Strategy 1 (l = {l})...")

        train_losses, val_losses, accuracies, train_time = train_model(model, train_loader_multi, val_loader_multi, criterion, optimizer, binary=False, epochs=10)

        print(f"Calculating test accuracy for multi-class classification (l = {l})...")
        acc, _ = calculate_accuracy(model, test_loader_multi, criterion, binary=False)
        print(f"Test accuracy: {acc * 100:.2f}%")

        print_losses(train_losses, val_losses, "S1_l=" + str(l))
        print_acc(accuracies, "S1_l=" + str(l))
        out_string = "l = " + str(l) + ": Final accuracy = " + str(acc) + ", training time = " + str(train_time) + " seconds"
        out_strings.append(out_string)

    l = 3
    model = copy.deepcopy(original_model)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 37))
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True)
    print('-' * 20)
    fine_tune_layers(model, l)
    print(f"Training for multi-class classification with Strategy 1 (l = {l}), higher lr")
    train_losses, val_losses, accuracies, train_time = train_model(model, train_loader_multi, val_loader_multi, criterion, optimizer, binary=False, epochs=10)
    print(f"Calculating test accuracy for multi-class classification (l = {l})...")
    acc, _ = calculate_accuracy(model, test_loader_multi, criterion, binary=False)
    print(f"Test accuracy: {acc * 100:.2f}%")
    print_losses(train_losses, val_losses, "S1_l=" + str(l) + ", higher lr")
    print_acc(accuracies, "S1_l=" + str(l) + ", higher lr")
        
    out_string = "l = " + str(l) + ", higher lr" + ": Final accuracy = " + str(acc) + ", training time = " + str(train_time) + " seconds"
    out_strings.append(out_string)
    model = copy.deepcopy(original_model)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 37))
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.01)
    print('-' * 20)
    fine_tune_layers(model, l)
    print(f"Training for multi-class classification with Strategy 1 (l = {l}), L2")
    train_losses, val_losses, accuracies, train_time = train_model(model, train_loader_multi, val_loader_multi, criterion, optimizer, binary=False, epochs=10)
    print(f"Calculating test accuracy for multi-class classification (l = {l})...")
    acc, _ = calculate_accuracy(model, test_loader_multi, criterion, binary=False)
    print(f"Test accuracy: {acc * 100:.2f}%")
    print_losses(train_losses, val_losses, "S1_l=" + str(l) + ", L2")
    print_acc(accuracies, "S1_l=" + str(l) + ", L2")
        
    out_string = "l = " + str(l) + ", L2" + ": Final accuracy = " + str(acc) + ", training time = " + str(train_time) + " seconds"
    out_strings.append(out_string)
    model = copy.deepcopy(original_model)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 37))
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    print('-' * 20)
    fine_tune_layers(model, l)
    print(f"Training for multi-class classification with Strategy 1 (l = {l}), new transform")
    train_losses, val_losses, accuracies, train_time = train_model(model, train_loader_multi_transform, val_loader_multi_transform, criterion, optimizer, binary=False, epochs=10)
    print(f"Calculating test accuracy for multi-class classification (l = {l})...")
    acc, _ = calculate_accuracy(model, test_loader_multi, criterion, binary=False)
    print(f"Test accuracy: {acc * 100:.2f}%")
    print_losses(train_losses, val_losses, "S1_l=" + str(l) + ", new transform")
    print_acc(accuracies, "S1_l=" + str(l) + ", new transform")
        
    out_string = "l = " + str(l) + ", new transform" + ": Final accuracy = " + str(acc) + ", training time = " + str(train_time) + " seconds"
    out_strings.append(out_string)

    with open("strategy_1.txt", "w") as f:
        for line in out_strings:
            f.write(line + "\n")

def strategy_two():
    out_strings = []

    for i in range(5):
        # Model setup
        model = copy.deepcopy(original_model)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 37)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        
        if i == 0:
            high_lr, reg, trans, diff = False, False, False, False
            train_losses, val_losses, accuracies, train_time = train_with_unfreezing(model, train_loader_multi, val_loader_multi, criterion, high_lr, reg, diff)
        elif i == 1:
            high_lr, reg, trans, diff = True, False, False, False
            train_losses, val_losses, accuracies, train_time = train_with_unfreezing(model, train_loader_multi, val_loader_multi, criterion, high_lr, reg, diff)
        elif i == 2:
            high_lr, reg, trans, diff = False, True, False, False
            train_losses, val_losses, accuracies, train_time = train_with_unfreezing(model, train_loader_multi, val_loader_multi, criterion, high_lr, reg, diff)
        elif i == 3:
            high_lr, reg, trans, diff = False, False, True, False
            train_losses, val_losses, accuracies, train_time = train_with_unfreezing(model, train_loader_multi_transform, val_loader_multi_transform, criterion, high_lr, reg, diff)
        else:
            high_lr, reg, trans, diff = False, False, False, True
            train_losses, val_losses, accuracies, train_time = train_with_unfreezing(model, train_loader_multi, val_loader_multi, criterion, high_lr, reg, diff)

        print("Evaluating on test set...")
        test_acc, test_loss = calculate_accuracy(model, test_loader_multi, criterion, False)
        print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

        print_losses(train_losses, val_losses, "Grad_unfreezing_high_lr_" + str(high_lr) + "_L2_" + str(reg) + "_trans_" + str(trans) + "_diff_" + str(diff))
        print_acc(accuracies, "Grad_unfreezing_high_lr_" + str(high_lr) + "_L2_" + str(reg) + "_trans_" + str(trans) + "_diff_" + str(diff))

        out_string = "high_lr: " + str(high_lr) + ", L2: " + str(reg) + ", trans: " + str(trans) + ", diff: " + str(diff) + ", Final accuracy = " + str(test_acc) + ", training time = " + str(train_time) + " seconds"
        out_strings.append(out_string)

    with open("strategy_2.txt", "w") as f:
        for line in out_strings:
            f.write(line + "\n")

def class_imbalance():
    layers = 5
    out_strings = []
    imbalance_criterion = nn.CrossEntropyLoss(weight = imbalanced_weights)
    for l in range(3, layers + 1):#range(3, layers + 1):
        train_losses = []
        val_losses = []
        accuracies = []

        model = copy.deepcopy(original_model)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_features, 37))
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)

        print('-' * 20)
        fine_tune_layers(model, l)
        print(f"Training for multi-class classification with imbalanced classes(l = {l})...")
        train_losses, val_losses, accuracies, train_time = train_model(model, train_loader_imbalance, val_loader_imbalance, imbalance_criterion, optimizer, binary=False, epochs=10)

        print(f"Calculating test accuracy for multi-class classification with imbalanced classes (l = {l})...")
        acc, _ = calculate_accuracy(model, test_loader_multi, imbalance_criterion, binary=False)
        print(f"Test accuracy: {acc * 100:.2f}%")

        print_losses(train_losses, val_losses, "imbal_l=" + str(l))
        print_acc(accuracies, "imbal_l=" + str(l))
        out_string = "l = " + str(l) + "final accuracy = " + str(acc) + ", training time = " + str(train_time) + " seconds"
        out_strings.append(out_string)
    with open("class_imbalance.txt", "w") as f:
        for line in out_strings:
            f.write(line + "\n")

def deep():
    results = []
    train_losses = []
    val_losses = []
    accuracies = []
    deep_model = models.resnet50(weights=deep_weight)
    criterion = nn.CrossEntropyLoss()
    optimizers = {'SGD', 'SGD_L2', 'AdamW'}
    
    for opt_name in optimizers:
        print(f"\nTraining ResNet50 with {opt_name} optimizer")
        
        fine_tune_depths = [0, 1, 3]
        for depth in fine_tune_depths:
            model = copy.deepcopy(deep_model)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Linear(num_features, 37))
            model = model.to(device)
            if opt_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
            elif opt_name == 'SGD_L2':
                optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.01)
            else:
                optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
            fine_tune_layers(model, depth)
            
            train_losses, val_losses, accuracies, train_time = train_model(
                model, train_loader_multi_deep, val_loader_multi_deep,
                criterion, optimizer, binary=False, epochs=10
            )
            
            test_acc, _ = calculate_accuracy(model, test_loader_multi_deep, criterion, binary=False)
            results.append({
                'optimizer': opt_name,
                'fine_tune_depth': depth,
                'test_accuracy': test_acc,
                'train_time': train_time
            })
            
            name = f"deep_{opt_name}_depth={depth}"
            print_losses(train_losses, val_losses, name)
            print_acc(accuracies, name)
    
    with open("deep.txt", "w") as f:
        for result in results:
            line = (f"Optim: {result['optimizer']}, "
                    f"Depth: {result['fine_tune_depth']}, "
                    f"Acc: {result['test_accuracy']:.4f}, "
                    f"Time: {result['train_time']}")
            f.write(line + "\n")

def batch_norm():
    

    # Model setup
    model = copy.deepcopy(original_model)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 37)
    model = model.to(device)
    freeze_all_but_fc(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters())


    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            optimizer.add_param_group({"params": module.parameters()})

    print(f"Training with unfrozen batch_norm layers")
    train_losses, val_losses, accuracies, train_time = train_model(model, train_loader_multi, val_loader_multi, criterion, optimizer, False, 10)

    print("Evaluating on test set...")
    test_acc, test_loss = calculate_accuracy(model, test_loader_multi, criterion, False)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    print_losses(train_losses, val_losses, "batch_norm_unfrozen")
    print_acc(accuracies, "batch_norm_unfrozen")

    #------------------------------
    # Model setup
    model = copy.deepcopy(original_model)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 37)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    freeze_batchnorm_stats(model)
    freeze_all_but_fc(model)
    optimizer = optim.Adam(model.fc.parameters())

    print(f"Training with frozen batch_norm layers")
    train_losses, val_losses, accuracies, train_time = train_model(model, train_loader_multi, val_loader_multi, criterion, optimizer, False, 10)

    print("Evaluating on test set...")
    test_acc, test_loss = calculate_accuracy(model, test_loader_multi, criterion, False)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    print_losses(train_losses, val_losses, "batch_norm_frozen")
    print_acc(accuracies, "batch_norm_frozen")





# binary()
# strategy_one()
# strategy_two()
# class_imbalance()
# deep()
batch_norm()
