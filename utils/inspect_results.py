# coding: utf-8
from IPython.display import display

from collections import defaultdict
import glob
import os
import json


def find_runs_on_filesystem(campaign_name, logs_filepath="../experiment-logs/", attach_rundirs=False):
    runs = []
    for run_dir in glob.glob("/".join([logs_filepath, "[0-9]*"])):
        run = {}
        try:
            with open(os.path.join(run_dir, "info.json"), "r") as f:
                run["info"] = json.load(f)
            with open(os.path.join(run_dir, "config.json"), "r") as f:
                run["config"] = json.load(f)
            with open(os.path.join(run_dir, "run.json"), "r") as f:
                run["run"] = json.load(f)

            if attach_rundirs:
                run["run_dir"] = run_dir

            if campaign_name:
                if run["config"]["experiment_name"] == campaign_name:
                    runs.append(run)
            else:
                runs.append(run)
        except IOError as e:
            pass
            # print(e)
    return runs


def list_campaigns(db_type):
    if db_type == "mongo":
        import pymongo
        client = pymongo.MongoClient("localhost", 27017)
        db = client.joint_ner_and_md
        campaign_names = db.runs.distinct('config.experiment_name')
    else:
        campaign_names = [x["config"]["experiment_name"] for x in find_runs_on_filesystem(None, logs_filepath=db_type)]
    return campaign_names


def report_results_of_a_specific_campaign(campaign_name, db_type):
    df = get_data_frame_for_results_of_a_specific_campaign(campaign_name, db_type)

    df_groupedby_hyperparameters = df.groupby(["integration_mode",
                                               "active_models",
                                               "use_golden_morpho_analysis_in_word_representation",
                                               "multilayer",
                                               "shortcut_connections",
                                               "lang_name"])
    return df, df_groupedby_hyperparameters.NER_best_test.mean()


def get_data_frame_for_results_of_a_specific_campaign(campaign_name, db_type, keys_to_report=None, extra_keys_to_report=None):
    if keys_to_report is None:
        keys_to_report = ["host",
                         "integration_mode",
                         "active_models",
                         "use_golden_morpho_analysis_in_word_representation",
                         "multilayer",
                         "shortcut_connections",
                         "epochs",
                         "lang_name",
                         "start_time",
                         "stop_time",
                         "duration"]
    if extra_keys_to_report is None:
        extra_keys_to_report = []
    keys_to_report += extra_keys_to_report
    print(campaign_name)
    runs = obtain_runs(campaign_name, db_type)
    configs = []
    for run_idx, run in enumerate(runs):

        dict_to_report = dict(run["config"])
        # u'start_time': u'2018-10-08T10:14:06.444095'
        import datetime
        if "stop_time" in run["run"]:
            stop_time_field_name = "stop_time"
        else:
            stop_time_field_name = "heartbeat"
        dict_to_report.update({run_field_name: datetime.datetime.strptime(run["run"][run_field_name], "%Y-%m-%dT%H:%M:%S.%f")
                               for run_field_name in ["start_time", stop_time_field_name]})
        dict_to_report["duration"] = dict_to_report[stop_time_field_name] - dict_to_report["start_time"]

        initial_keys = list(dict_to_report.keys())

        print(initial_keys)

        result_designation_labels = ["MORPH", "NER"]

        dict_to_report["epochs"] = max([len(list(run["info"][label].keys()))
                                        for label in ["NER_dev_f_score", "MORPH_dev_f_score"]])

        for result_designation_label in result_designation_labels:

            print("result_designation_label: ", result_designation_label)

            best_performances = run["info"][result_designation_label + "_dev_f_score"]
            print(best_performances)
            best_dev_result_for_this_run = 0
            best_test_result_for_this_run = 0
            epoch_id_of_the_best_dev_result = -1
            # display(run["config"])
            for epoch in sorted([int(k) for k in list(best_performances.keys())]):
                # if result_designation_label != "NER":
                #     corrected_epoch = epoch + 1
                epoch_max = max(best_performances[str(epoch)])
                if epoch_max > best_dev_result_for_this_run:
                    epoch_id_of_the_best_dev_result = epoch
                    best_dev_result_for_this_run = epoch_max
                    best_test_result_for_this_run = \
                        max(run["info"][result_designation_label + "_test_f_score"][str(epoch)])

                # print "run_idx: %d, epoch: %d, epoch_best_performance: %.2lf, best_for_this_run: %.2lf" % (run_idx, epoch, epoch_max, best_for_this_run)

            dict_to_report[result_designation_label + "_best_dev"] = best_dev_result_for_this_run
            dict_to_report[result_designation_label + "_best_test"] = best_test_result_for_this_run

            for x in result_designation_labels:
                # if x != result_designation_label:
                print("x: ", x)
                print("epoch_id_of_the_best_dev_result: ", epoch_id_of_the_best_dev_result)
                dict_to_report[result_designation_label + "_to_" + x + "_test"] = \
                    max(run["info"][x + "_test_f_score"][str(epoch_id_of_the_best_dev_result)]) \
                        if str(epoch_id_of_the_best_dev_result) in list(run["info"][x + "_test_f_score"].keys()) else -1
                print(dict_to_report[result_designation_label + "_to_" + x + "_test"])

        for key in run["info"].keys():
            if key.startswith("NER_TYPE_"):
                best_dev_result_for_this_run = 0
                f_scores = run["info"][key]
                for epoch in sorted([int(k) for k in list(f_scores.keys())]):
                    epoch_max = max([float(x) for x in f_scores[str(epoch)]])
                    if epoch_max > best_dev_result_for_this_run:
                        epoch_id_of_the_best_dev_result = epoch
                        best_dev_result_for_this_run = epoch_max
                dict_to_report[key+ "_best"] = best_dev_result_for_this_run

        configs.append({key: dict_to_report[key]
                        for key in
                        [x for x in keys_to_report if x in dict_to_report] +
                        [x for x in list(dict_to_report.keys()) if x not in initial_keys] +
                        [x for x in list(dict_to_report.keys()) if x.startswith("NER_TYPE")]})
    import pandas
    df = pandas.DataFrame.from_dict(configs)
    print(configs)
    cols = df.columns.tolist()
    # display(df[["host"] +
    #                     [x for x in dict_to_report.keys() if x not in initial_keys]])
    display(df)
    return df


def generate_df_for_losses(campaign_name, db_type="../experiment-logs/"):
    runs = obtain_runs(campaign_name = campaign_name, db_type = db_type)
    configs = []
    initial_keys = ['']
    for run_idx, run in enumerate(runs):
        dict_to_report = dict(run["config"])
        initial_keys = list(dict_to_report.keys())

        # add duration field
        import datetime
        if "stop_time" in run["run"]:
            stop_time_field_name = "stop_time"
        else:
            stop_time_field_name = "heartbeat"
        dict_to_report.update(
            {run_field_name: datetime.datetime.strptime(run["run"][run_field_name], "%Y-%m-%dT%H:%M:%S.%f")
             for run_field_name in ["start_time", stop_time_field_name]})
        dict_to_report["duration"] = dict_to_report[stop_time_field_name] - dict_to_report["start_time"]

        # add epochs field
        dict_to_report["epochs"] = max([len(list(run["info"][label].keys()))
                                        for label in ["NER_dev_f_score", "MORPH_dev_f_score"]])

        dict_to_report["avg_losses"] = [float(x[1][-1]) for x in sorted(run["info"]["avg_loss"].items(), key=lambda x: x[0])] if "avg_loss" in run["info"] else []

        first_part_of_fields = [
            "host",
            "integration_mode",
            "active_models",
            "use_golden_morpho_analysis_in_word_representation",
            "multilayer",
            "shortcut_connections",
            "epochs",
            "lang_name",
            "start_time",
            "stop_time",
            "duration"]
        extra_fields = [x for x in list(dict_to_report.keys()) if x not in initial_keys]
        # filter out unwanted fields from the reported row
        configs.append({key: dict_to_report[key]
                        for key in
                        [x for x in first_part_of_fields if x in dict_to_report] +
                        extra_fields})
    import pandas
    df = pandas.DataFrame.from_dict(configs)
    print(configs)
    cols = df.columns.tolist()
    # display(df[["host"] +
    #                     [x for x in dict_to_report.keys() if x not in initial_keys]])
    display(df)
    return df


def obtain_runs(campaign_name, db_type):
    if db_type == "mongo":
        import pymongo
        client = pymongo.MongoClient("localhost", 27017)
        db = client.joint_ner_and_md
        runs = db.runs.find({"config.experiment_name": campaign_name})
    else:
        runs = find_runs_on_filesystem(campaign_name, logs_filepath=db_type)
    return runs


def report(campaign_name="TRUBA-20181010-over-all-languages-03-dim-10-morpho_tag_type-char"):
    df6, _ = report_results_of_a_specific_campaign(
        campaign_name=campaign_name, db_type = "../experiment-logs/")

    # TODO: find a way to make 0 epochs 1 or ignore.
    df7 = df6.assign(duration_per_epoch=lambda x: x.duration / (x.epochs + 1))

    for lang_name in "czech spanish hungarian finnish turkish".split(" "):
        df_lang_name = df7.loc[df7['lang_name'] == lang_name].sort_values(["integration_mode",
                                                                           "active_models",
                                                                           "use_golden_morpho_analysis_in_word_representation",
                                                                           "multilayer",
                                                                           "shortcut_connections",
                                                                           "lang_name"])
        print(lang_name)
        print("duration_per_epoch", df_lang_name.duration_per_epoch.mean())
        print("epochs", df_lang_name.epochs.mean())
        print("epochs", df_lang_name.epochs.describe())
        print("NER_best_test", df_lang_name.groupby(["integration_mode",
                                                     "active_models",
                                                     "use_golden_morpho_analysis_in_word_representation",
                                                     "multilayer",
                                                     "shortcut_connections",
                                                     "lang_name"]).NER_best_test.mean())

        df_lang_name.to_csv("./reports/report-%s-%s.csv" % (campaign_name, lang_name))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--command", choices=["create_report", "grep_logdirs", "print_resumable_experiment_configurations"], required=True)

    parser.add_argument("--campaign_name", default="section1-all-20171013-01")

    parser.add_argument("--db_type", default="mongo")

    args = parser.parse_args()

    if args.command == "create_report":

        df, df_groupedby_hyperparameter_NER_best_test_mean = report_results_of_a_specific_campaign(args.campaign_name,
                                                                                                   args.db_type)
        df.to_csv("./reports/results-%s.csv" % args.campaign_name)
        df_groupedby_hyperparameter_NER_best_test_mean.to_csv(
            "./reports/results-NER_best_test_mean-%s.csv" % args.campaign_name)

    elif args.command == "grep_logdirs":
        runs = find_runs_on_filesystem(args.campaign_name, logs_filepath=args.db_type, attach_rundirs=True)
        for run in runs:
            output_line = [run["run_dir"]]
            if "model_dir_path" in run["info"]:
                output_line.append(run["info"]["model_dir_path"])
            else:
                output_line.append("")
            if "model_epoch_dir_path" in run["info"]:
                output_line.append(os.path.join(run["info"]["model_dir_path"], run["info"]["model_epoch_dir_path"]))
            else:
                output_line.append("")
            sorted_epoch_nos = sorted([int(x) for x in run["info"]["avg_loss"].keys()])
            if sorted_epoch_nos:
                output_line.append(str(sorted_epoch_nos[-1]))
            else:
                output_line.append("1")
            print(" ".join(output_line))
    elif args.command == "print_resumable_experiment_configurations":
        runs = find_runs_on_filesystem(args.campaign_name, logs_filepath=args.db_type, attach_rundirs=True)
        output_lines = defaultdict(list)
        for run in runs:
            # print(run["run_dir"])
            reload = 1
            output_line = []
            if "model_dir_path" in run["info"]:
                output_line.append("=".join(["model_path", run["info"]["model_dir_path"]]))
            else:
                reload = 0
            if "model_epoch_dir_path" in run["info"]:
                output_line.append("=".join(["model_epoch_path",
                                             os.path.join(run["info"]["model_dir_path"], run["info"]["model_epoch_dir_path"])]))
            else:
                reload = 0
            if reload == 0:
                break
            else:
                output_line.append("reload=%d" % reload)
                sorted_epoch_nos = sorted([int(x) for x in run["info"]["avg_loss"].keys()])
                if sorted_epoch_nos:
                    output_line.append("=".join(["starting_epoch_no", str(sorted_epoch_nos[-1])]))
                else:
                    output_line.append("starting_epoch_no=1")

                parameters_to_be_ignored = "debug maximum_epochs reload model_path model_epoch_path starting_epoch_no".split(" ")

                output_line += [(key+"="+value) for key,value in [x.split("=") for x in run["run"]["meta"]["options"]["UPDATE"]] if key not in parameters_to_be_ignored]

                """
                model_epoch_path=./models/model-00000234/model-epoch-00000001 starting_epoch_no=11
                """
                output_line_dict = {key: value for key, value in [x.split("=") for x in output_line]}
                output_line_key = " ".join(sorted([(key+"="+value) for key, value in [x.split("=") for x in output_line] if key not in ["model_epoch_path", "starting_epoch_no"]]))
                output_lines[output_line_key].append([int(output_line_dict["starting_epoch_no"]), output_line])

        for output_line_key in output_lines.keys():
            output_line_tuples = output_lines[output_line_key]

            output_line = sorted(output_line_tuples, key=lambda x: x[0], reverse=True)[0]

            print(" ".join(output_line[1]))

