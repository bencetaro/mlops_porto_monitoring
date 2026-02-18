import json, os
import matplotlib.pyplot as plt

def write_json(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

def plot_mutual_info(mi_df, output_path):
    plt.figure(figsize=(10, 6))
    plt.barh(mi_df["feature"], mi_df["mi_score"])
    plt.xlabel("MI Score")
    plt.title("Feature Importance based on MI")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_feature_importances(model_name, fi_df, output_path):
    plt.figure(figsize=(10, 6))
    plt.barh(fi_df["feature"], fi_df["importance"])
    plt.xlabel("Feature Importance")
    plt.title(f"Feature Importance of {model_name} model")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(cm, output_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(cm))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def update_symlink(output_dir, model_filename):
    link_path = os.path.join(output_dir, "model.pkl")
    target_path = os.path.join(output_dir, "models", model_filename)

    if os.path.islink(link_path) or os.path.exists(link_path):
        os.remove(link_path)
    os.symlink(target_path, link_path) # create symlink
