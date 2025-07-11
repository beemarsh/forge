import argparse
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

def get_path(input_path, output_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isabs(input_path):
        absolute_input_path = os.path.join(script_dir, input_path)
    else:
        absolute_input_path = input_path
    normalized_absolute_input_path = os.path.normpath(absolute_input_path)

    if not os.path.isabs(output_path):
        absolute_output_dir = os.path.join(script_dir, output_path)
    else:
        absolute_output_dir = output_path
    normalized_absolute_output_dir = os.path.normpath(absolute_output_dir)

    # Check if input path exists and is a directory
    if not os.path.exists(normalized_absolute_input_path) or not os.path.isdir(normalized_absolute_input_path):
        raise FileNotFoundError(f"Input directory does not exist: {normalized_absolute_input_path}")

    # Check if output path exists
    if os.path.exists(normalized_absolute_output_dir):
        raise FileExistsError(f"Output directory already exists: {normalized_absolute_output_dir}")
    else:
        # Create the output directory
        os.makedirs(normalized_absolute_output_dir)

    return normalized_absolute_input_path, normalized_absolute_output_dir

def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

def main():

    parser = argparse.ArgumentParser(
        description="Process a path argument and convert it to an absolute path."
    )
    parser.add_argument(
        '--tensorboard-dir',
        type=str,
        required=True,
        help='Directory holding the tensorboard file generated by FORGE(can be relative or absolute).'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='The directory where output files will be saved.'
    )
    
    args = parser.parse_args()
    input_path, output_path = get_path(args.tensorboard_dir, args.output_dir)

    df = tflog2pandas(input_path)
    metrics = df['metric'].unique()

    for metric in metrics:
        n_df = df[df["metric"] == metric].copy()
        n_df.drop('metric', axis=1, inplace=True)
        file_name = metric.replace("/","_")
        n_df.to_csv(f'{output_path}/{file_name}.csv', index=False)
        # print(metric_name)
        # print(n_df)
    
if __name__ == "__main__":
    main()
