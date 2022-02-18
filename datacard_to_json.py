import json
import logging
import sys
from copy import deepcopy
from typing import Any
import argparse

log = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert datacard to JSON")

    parser.add_argument("--datacard", required=True, type=str)

    parser.add_argument("--mass", type=str, default="125")

    return parser.parse_args()


def json_str(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=4)


def restructure_shapes(
    shape_lines: list, bins: list, processes: list, systematics: list, mass: str
) -> list:
    processes.append("data_obs")  # NBBBB!!!!!!!
    shape_lines_reco = []
    shape_dict_list = []
    identifiers = [
        "type",
        "process",
        "bin",
        "file",
        "histogram",
        "histogram_with_systematics",
    ]
    for shape_line in shape_lines:
        line_dict = {}
        for id, word in zip(identifiers, shape_line.split()):
            line_dict[id] = word

        shape_lines_reco.append(line_dict)

    # log.debug("shape_lines_reco:\n{}".format(shape_lines_reco))
    for shape_line in shape_lines_reco:
        # log.debug(shape_line)
        # main
        if shape_line["process"] == "*":
            # remove processes that are already in other lines
            others = []
            for sl in shape_lines_reco:
                if sl["process"] != "*":
                    others.append(sl["process"])
            l_processes = [process for process in processes if process not in others]
        else:
            l_processes = [shape_line["process"]]
        l_processes = list(set(l_processes))
        log.debug("l_processes = {}".format(l_processes))
        if shape_line["bin"] == "*":
            # remove bins that are already in other lines
            others = []
            for sl in shape_lines_reco:
                if sl["bin"] != "*":
                    others.append(sl["bin"])
            l_bins = [bin for bin in bins if bin not in others]
        else:
            l_bins = [shape_line["bin"]]
        l_bins = list(set(l_bins))
        # log.debug("l_bins = {}".format(l_bins))

        for bin in l_bins:
            for process in l_processes:
                shape_dict_list.append(
                    {
                        "process": process,
                        "bin": bin,
                        "file": shape_line["file"],
                        "histogram": shape_line["histogram"]
                        .replace("$bin", bin)
                        .replace("$MASS", mass)
                        .replace("$PROCESS", process),
                    }
                )

        # systematics
        if "histogram_with_systematics" in shape_line:
            # log.debug("Now creating systematics for shape_line \n{}".format(shape_line))
            for sys in systematics:
                for bin in l_bins:
                    for process in l_processes:
                        # log.debug("shape_line:\n{}\nsys: {}, bin: {}, process: {}".format(shape_line, sys, bin, process))
                        shape_dict_list.append(
                            {
                                "process": process,
                                "bin": bin,
                                "file": shape_line["file"],
                                "modifier": sys,
                                "histogram_up": "{}Up".format(
                                    shape_line["histogram_with_systematics"]
                                    .replace("$bin", bin)
                                    .replace("$MASS", mass)
                                    .replace("$PROCESS", process)
                                    .replace("$SYSTEMATIC", sys)
                                ),
                                "histogram_down": "{}Down".format(
                                    shape_line["histogram_with_systematics"]
                                    .replace("$bin", bin)
                                    .replace("$MASS", mass)
                                    .replace("$PROCESS", process)
                                    .replace("$SYSTEMATIC", sys)
                                ),
                            }
                        )

    # log.debug("shapes dict: \n{}".format(json_str(shape_dict_list)))

    return shape_dict_list


def restructure_observations(observations: list, shapes_list: list) -> list:
    """build list of dictionaries with observed yields"""
    obs_dict_list = []
    # not clear how multi-bin bins are specified here
    for i_ch, (bin, rate) in enumerate(
        zip(observations[0].split()[1:], observations[1].split()[1:])
    ):
        obs = float(observations[1].split()[1:][i_ch])  # could use int for data
        dct = {"name": bin, "rate": obs}
        for shape_dict in shapes_list:
            if (
                shape_dict["process"] == "data_obs"
                and shape_dict["bin"] == bin
                and "modifier" not in shape_dict
            ):
                log.debug("Found it!")
                dct["data"] = f"{shape_dict['file']}:{shape_dict['histogram']}"
        obs_dict_list.append(dct)
    log.debug(
        f"\nobs dict (after restructuring observations):\n{json_str(obs_dict_list)}\n"
    )
    return obs_dict_list


def restructure_bins(processes: list) -> list:
    """build list of bins with processes and their observed yields"""
    ch_dict_list = []
    # this assumes order bin - process - process - rate
    # loop over bins
    bin_names = processes[0].split()[1:]
    process_names = processes[1].split()[1:]
    processes_numbers = processes[2].split()[1:]
    yields = [float(y) for y in processes[3].split()[1:]]
    log.debug("yields: {}".format(yields))
    # loop over bins
    for ch in sorted(set(bin_names)):
        # get indices of current bin
        ch_idx = [i for i, c in enumerate(bin_names) if c == ch]
        process_dict_list = []
        for i_sam, (process, process_num, rate) in enumerate(
            zip(
                [process_names[i] for i in ch_idx],
                [processes_numbers[i] for i in ch_idx],
                [yields[i] for i in ch_idx],
            )
        ):
            # include a placeholder for process modifiers
            process_dict_list.append(
                {
                    "name": process,
                    "id": process_num,
                    "rate": rate,
                    "data": [yields[ch_idx[i_sam]]],
                    "modifiers": [],
                }
            )
        ch_dict_list.append({"name": ch, "processes": process_dict_list})
    # log.debug(f"\nch dict:\n{json_str(ch_dict_list)}\n")
    return ch_dict_list


def restructure_modifiers(
    modifiers: list,
    bin_names: list,
    bin_yields: list,
    process_names: list,
    process_numbers: list,
    shape_identifiers: list = [],
) -> dict:
    """build a dictionary with modifiers per process from datacard"""
    n_processes = len(process_names)

    # placeholder collecting modifiers per bin and process, list of dicts
    # example: modifier_dict[ch_name][sam_name] is list of modifiers
    modifier_dict = {}
    for ch in bin_names:
        modifier_dict.update({ch: {}})
        for s in process_names:
            modifier_dict[ch].update({s: []})

    for line in modifiers:
        # parse each modifier
        line_split = line.split()
        syst_name = line_split[0]
        syst_type = line_split[1]
        if syst_type == "gmN":
            # additional entry needed for gammas
            # currently unclear how extrapolation factor enters
            n_evts_CR = int(line_split[2])
            # see https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/part2/settinguptheanalysis/#a-simple-counting-experiment
            stat_unc = 1 / (1 + n_evts_CR) ** 0.5
            norm_effects = line_split[3 : 3 + n_processes]
            # override extrapolation factors with rel. stat unc
            norm_effects = [stat_unc if n != "-" else n for n in norm_effects]
        else:
            norm_effects = line_split[2 : 2 + n_processes]
        norm_effects = [float(n) if n != "-" else 0.0 for n in norm_effects]
        # log.debug(f"syst {syst_name} with type {syst_type} and effects {norm_effects}")

        for i, norm_effect in enumerate(norm_effects):
            # go through each process affected by a modifier
            if norm_effect == 0.0:
                continue  # no effect, skip
            bin_name = bin_names[i]
            process_name = process_names[i]
            process_number = process_numbers[i]
            # log.debug(
            #    f" - norm effect {norm_effect} for {process_name} in {bin_name}"
            # )
            if syst_type == "lnN":
                if isinstance(norm_effect, float):
                    modifier_dict[bin_name][process_name].append(
                        {
                            "name": syst_name,
                            "type": "normsys",
                            "data": {"hi": norm_effect, "lo": 1 / norm_effect},
                        }
                    )
                elif isinstance(norm_effect, str):
                    lo, hi = norm_effect.split("/")
                    modifier_dict[bin_name][process_name].append(
                        {
                            "name": syst_name,
                            "type": "normsys",
                            "data": {"hi": float(hi), "lo": float(lo)},
                        }
                    )
            elif syst_type == "lnU":
                # don't know exactly how this should be treated;
                # have it like this for now, then we'll see
                modifier_dict[bin_name][process_name].append(
                    {
                        "name": syst_name,
                        "type": "normsys - lnU",
                        "data": {"hi": norm_effect, "lo": 1 / norm_effect},
                    }
                )
            elif syst_type == "gmN":
                # this needs access to bin yields to calculate absolute stat. unc.
                abs_stat_unc = norm_effect * bin_yields[i]
                modifier_dict[bin_name][process_name].append(
                    {"name": syst_name, "type": "staterror", "data": [abs_stat_unc]}
                )
            elif (
                syst_type == "shape" or syst_type == "shape?"
            ):  # still have to understand what exactly shape? means
                # see chat with ACM for meaning of shape?
                # first we scan shape_identifiers to find the modifier for this bin/process combination
                for shape_identifier in shape_identifiers:
                    if (
                        "modifier" in shape_identifier
                        and shape_identifier["process"] == process_name
                        and shape_identifier["bin"] == bin_name
                        and shape_identifier["modifier"] == syst_name
                    ):
                        histo_up = f"{shape_identifier['file']}:{shape_identifier['histogram_up']}"
                        histo_down = f"{shape_identifier['file']}:{shape_identifier['histogram_down']}"
                        modifier_dict[bin_name][process_name].append(
                            {
                                "name": syst_name,
                                "type": "shaperror",
                                "data": norm_effect,
                                "shape": {"up": histo_up, "down": histo_down},
                            }
                        )
                        break
            else:
                raise NotImplementedError(
                    "syst_type {} not supported".format(syst_type)
                )
    # log.debug(f"\nmodifier dict:\n{json_str(modifier_dict)}\n")
    return modifier_dict


def get_sections_dict(datacard: list, mass: str) -> dict:
    """extract info from datacard into dictionary"""
    sections_list = []
    current_section = []
    for line in datacard:
        line_stripped = line.strip()
        if len(line_stripped) == 0 or (
            len(line_stripped) > 0 and line_stripped[0] == "#"
        ):
            # skip comments and empty lines
            continue
        if line_stripped[0] == "-":
            # end of section
            sections_list.append(current_section)
            current_section = []
        else:
            current_section.append(line_stripped)
        if line == datacard[-1]:
            # append last section
            sections_list.append(current_section)

    sections_dict = {}
    # find "general" section with imax etc.
    # seems to be first usually
    # not clear yet that this is needed
    sections_dict.update({"general": sections_list.pop(0)})

    # root files for shapes analyses
    try:
        idx = next(
            i
            for i, s in enumerate(sections_list)
            if any([l.startswith("shapes") for l in s])
        )
        sections_dict.update({"shapes": sections_list.pop(idx)})
    except StopIteration:
        pass

    # data yields, identified by "observation"
    idx = next(
        i
        for i, s in enumerate(sections_list)
        if any(["observation" in l[0:11] for l in s])
    )
    sections_dict.update({"observations": sections_list.pop(idx)})
    log.debug(f"observations: {sections_dict['observations']}")
    # process yields, identified by "rate"
    idx = next(
        i for i, s in enumerate(sections_list) if any(["rate" in l[0:4] for l in s])
    )
    sections_dict.update({"bins": sections_list.pop(idx)})

    # UPDATED: systematics, last in list (need better identifier)
    # in the shapes configuration, last lines can be of format:
    # <something> autoMCStats <many something>
    # <something> rateParam <many something>
    # we exclude them and try to understand what they do later on
    modifiers = []
    unsupported_modifiers = ["autoMCStats", "rateParam", "param"]
    for line in sections_list[-1]:
        if any(
            unsupported_modifier in line
            for unsupported_modifier in unsupported_modifiers
        ):
            continue
        else:
            modifiers.append(line)
    sections_dict.update({"modifiers": modifiers})
    # log.debug(f"modifiers: {sections_dict['modifiers']}")

    # full list of bins and processes from datacard (including duplications)
    bin_names = sections_dict["bins"][0].split()[1:]
    bin_yields = [float(y) for y in sections_dict["bins"][3].split()[1:]]
    process_names = sections_dict["bins"][1].split()[1:]
    process_numbers = sections_dict["bins"][2].split()[1:]
    modifier_names = [line.split()[0] for line in sections_dict["modifiers"]]
    log.debug("Modifiers: {}".format(modifier_names))

    # restructure shapes
    if "shapes" in sections_dict:
        sections_dict["shapes"] = restructure_shapes(
            sections_dict["shapes"], bin_names, process_names, modifier_names, mass
        )
    else:
        sections_dict["shapes"] = []

    # convert observations into dict
    sections_dict["observations"] = restructure_observations(
        sections_dict["observations"], sections_dict["shapes"]
    )

    # convert bin information (process yields) into dict
    sections_dict["bins"] = restructure_bins(sections_dict["bins"])

    # convert modifier information into dict
    # needs access to full lists of bins (+ yields) and process names
    sections_dict["modifiers"] = restructure_modifiers(
        sections_dict["modifiers"],
        bin_names,
        bin_yields,
        process_names,
        process_numbers,
        sections_dict["shapes"],
    )

    # log.debug("sections_dict = {}".format(sections_dict))

    return sections_dict


def sections_dict_to_workspace(sections_dict: dict) -> dict:
    """convert dictionary with info from datacard into workspace"""
    ws = {}
    # need to add signal POI manually, assuming signal is first process
    for bin in sections_dict["bins"]:
        for i, process in enumerate(bin["processes"]):
            # attach modifiers to this process for this bin
            process["modifiers"] = sections_dict["modifiers"][bin["name"]][
                process["name"]
            ]
            # attach normfactor for signals
            # signals are identified by an id <= 0
            idn = int(process["id"])
            if idn <= 0:
                # this should be signal, attach normfactor
                process["modifiers"].append(
                    {
                        "data": None,
                        "name": "r_{}".format(process["name"]),
                        "type": "normfactor",
                    }
                )
            # time to add shapes, if shapes analysis
            if sections_dict["shapes"]:
                # loop to check which one matches all the requirements
                for shape_dict in sections_dict["shapes"]:
                    if (
                        "modifier" not in shape_dict
                        and shape_dict["process"] == process["name"]
                        and shape_dict["bin"] == bin["name"]
                    ):
                        process[
                            "data"
                        ] = f"{shape_dict['file']}:{shape_dict['histogram']}"
                        break

    ws.update({"bins": sections_dict["bins"]})
    ws.update(
        {"measurements": [{"config": {"parameters": [], "poi": "r"}, "name": "meas"}]}
    )
    ws.update({"observations": sections_dict["observations"]})
    ws.update({"version": "1.0.0"})

    return ws


def datacard_to_json(datacard: list, mass: str) -> dict:
    sections_dict = get_sections_dict(datacard, mass)
    ws = sections_dict_to_workspace(sections_dict)
    # log.debug(f"HistFactory workspace in JSON:\n{json_str(ws)}\n")
    return ws


if __name__ == "__main__":
    args = parse_arguments()
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")

    with open(args.datacard) as f:
        datacard = f.readlines()

    ws = datacard_to_json(datacard, args.mass)
    ws_name = ".".join(args.datacard.split(".")[0:-1]) + ".json"

    log.info(f"saving workspace as {ws_name}")
    with open(ws_name, "w") as f:
        f.write(json_str(ws))
