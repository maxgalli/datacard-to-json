import json
import logging
import sys
import uproot
from copy import deepcopy
from typing import Any


log = logging.getLogger(__name__)


def json_str(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=4)


def restructure_shapes(shape_lines: list, channels: list, samples: list, systematics: list, mass: str) -> list:
    samples.append("data_obs") # NBBBB!!!!!!!
    shape_lines_reco = []
    shape_dict_list = []
    identifiers = [
        "type", "sample", "channel", "file", "histogram", "histogram_with_systematics"
    ]
    for shape_line in shape_lines:
        line_dict = {}
        for id, word in zip(identifiers, shape_line.split()):
            line_dict[id] = word
        
        shape_lines_reco.append(line_dict)

    #log.debug("shape_lines_reco:\n{}".format(shape_lines_reco))
    for shape_line in shape_lines_reco:
        #log.debug(shape_line)
        # main
        if shape_line["sample"] == "*":
            # remove samples that are already in other lines
            others = []
            for sl in shape_lines_reco:
                if sl["sample"] != "*":
                    others.append(sl["sample"])
            l_samples = [sample for sample in samples if sample not in others]
        else:
            l_samples = [shape_line["sample"]]
        l_samples = list(set(l_samples))
        log.debug("l_samples = {}".format(l_samples))
        if shape_line["channel"] == "*":
            # remove channels that are already in other lines
            others = []
            for sl in shape_lines_reco:
                if sl["channel"] != "*":
                    others.append(sl["channel"])
            l_channels = [channel for channel in channels if channel not in others]
        else:
            l_channels = [shape_line["channel"]]
        l_channels = list(set(l_channels))
        #log.debug("l_channels = {}".format(l_channels))

        for channel in l_channels:
            for sample in l_samples:
                shape_dict_list.append(
                    {
                        "sample": sample,
                        "channel": channel,
                        "file": shape_line["file"],
                        "histogram": "{}/{}".format(channel, shape_line["histogram"].replace("$MASS", mass).replace("$PROCESS", sample))
                    }
                )

        # systematics
        if "histogram_with_systematics" in shape_line:
            #log.debug("Now creating systematics for shape_line \n{}".format(shape_line))
            for sys in systematics:
                for channel in l_channels:
                    for sample in l_samples:
                        #log.debug("shape_line:\n{}\nsys: {}, channel: {}, sample: {}".format(shape_line, sys, channel, sample))
                        shape_dict_list.append(
                            {
                                "sample": sample,
                                "channel": channel,
                                "file": shape_line["file"],
                                "modifier": sys,
                                "histogram_up": "{}/{}Up".format(channel, 
                                shape_line["histogram_with_systematics"].replace("$MASS", mass).replace("$PROCESS", sample).replace("$SYSTEMATIC", sys)
                                    ),
                                "histogram_down": "{}/{}Down".format(channel, 
                                shape_line["histogram_with_systematics"].replace("$MASS", mass).replace("$PROCESS", sample).replace("$SYSTEMATIC", sys)
                                    ),
                            }
                        )                      

    #log.debug("shapes dict: \n{}".format(json_str(shape_dict_list)))
    
    return shape_dict_list


def restructure_observations(observations: list, shapes_list: list) -> list:
    """build list of dictionaries with observed yields"""
    obs_dict_list = []
    # not clear how multi-bin channels are specified here
    for i_ch, channel in enumerate(observations[0].split()[1:]):
        obs = float(observations[1].split()[1:][i_ch])  # could use int for data
        dct = {"data": [obs], "name": channel}
        for shape_dict in shapes_list:
            if shape_dict["sample"] == "data_obs" and shape_dict["channel"] == channel and "modifier" not in shape_dict:
                log.debug("Found it!") 
                dct["shape"] = extract_histogram(shape_dict["file"], shape_dict["histogram"])
        obs_dict_list.append(dct)
    #log.debug(f"\nobs dict (after restructuring observations):\n{json_str(obs_dict_list)}\n")
    return obs_dict_list


def restructure_channels(samples: list) -> list:
    """build list of channels with samples and their observed yields"""
    ch_dict_list = []
    # this assumes order bin - process - process - rate
    # loop over channels
    channel_names = samples[0].split()[1:]
    sample_names = samples[1].split()[1:]
    samples_numbers = samples[2].split()[1:]
    yields = [float(y) for y in samples[3].split()[1:]]
    # loop over channels
    for ch in sorted(set(channel_names)):
        # get indices of current channel
        ch_idx = [i for i, c in enumerate(channel_names) if c == ch]
        sample_dict_list = []
        for i_sam, (sample, sample_num) in enumerate(zip(
            [sample_names[i] for i in ch_idx], 
            [samples_numbers[i] for i in ch_idx])
            ):
            # include a placeholder for sample modifiers
            sample_dict_list.append(
                {
                    "name": sample,
                    "id": sample_num, 
                    "data": [yields[ch_idx[i_sam]]], 
                    "modifiers": []}
            )
        ch_dict_list.append({"name": ch, "samples": sample_dict_list})
    #log.debug(f"\nch dict:\n{json_str(ch_dict_list)}\n")
    return ch_dict_list


def restructure_modifiers(
    modifiers: list, channel_names: list, channel_yields: list, sample_names: list, sample_numbers: list, shape_identifiers: list = []
) -> dict:
    """build a dictionary with modifiers per sample from datacard"""
    n_processes = len(sample_names)

    # placeholder collecting modifiers per channel and sample, list of dicts
    # example: modifier_dict[ch_name][sam_name] is list of modifiers
    modifier_dict = {}
    for ch in channel_names:
        modifier_dict.update({ch: {}})
        for s in sample_names:
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
        #log.debug(f"syst {syst_name} with type {syst_type} and effects {norm_effects}")

        for i, norm_effect in enumerate(norm_effects):
            # go through each sample affected by a modifier
            if norm_effect == 0.0:
                continue  # no effect, skip
            channel_name = channel_names[i]
            sample_name = sample_names[i]
            sample_number = sample_numbers[i]
            #log.debug(
            #    f" - norm effect {norm_effect} for {sample_name} in {channel_name}"
            #)
            if syst_type == "lnN":
                if isinstance(norm_effect, float):
                    modifier_dict[channel_name][sample_name].append(
                        {
                            "name": syst_name,
                            "type": "normsys",
                            "data": {"hi": norm_effect, "lo": 1 / norm_effect},
                        }
                    )
                elif isinstance(norm_effect, str):
                    lo, hi = norm_effect.split("/")
                    modifier_dict[channel_name][sample_name].append(
                        {
                            "name": syst_name,
                            "type": "normsys",
                            "data": {"hi": float(hi), "lo": float(lo)},
                        }
                    )
            elif syst_type == "lnU":
                # don't know exactly how this should be treated;
                # have it like this for now, then we'll see
                    modifier_dict[channel_name][sample_name].append(
                        {
                            "name": syst_name,
                            "type": "normsys - lnU",
                            "data": {"hi": norm_effect, "lo": 1 / norm_effect},
                        }
                    )               
            elif syst_type == "gmN":
                # this needs access to channel yields to calculate absolute stat. unc.
                abs_stat_unc = norm_effect * channel_yields[i]
                modifier_dict[channel_name][sample_name].append(
                    {"name": syst_name, "type": "staterror", "data": [abs_stat_unc]}
                )
            elif syst_type == "shape" or syst_type == "shape?": # still have to understand what exactly shape? means
                # see chat with ACM for meaning of shape?
                # first we scan shape_identifiers to find the modifier for this channel/sample combination
                for shape_identifier in shape_identifiers:
                    if "modifier" in shape_identifier and shape_identifier["sample"] == sample_name \
                        and shape_identifier["channel"] == channel_name and shape_identifier["modifier"] == syst_name:
                        histo_up = extract_histogram(shape_identifier["file"], shape_identifier["histogram_up"])
                        histo_down = extract_histogram(shape_identifier["file"], shape_identifier["histogram_down"])
                        modifier_dict[channel_name][sample_name].append(
                            {
                                "name": syst_name,
                                "type": "shaperror",
                                "data": norm_effect,
                                "shape": {"up": histo_up, "down": histo_down}
                            }
                        )
                        break
            else:
                raise NotImplementedError("syst_type {} not supported".format(syst_type))
    #log.debug(f"\nmodifier dict:\n{json_str(modifier_dict)}\n")
    return modifier_dict


def get_sections_dict(datacard: list) -> dict:
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
            i for i, s in enumerate(sections_list) if any([l.startswith("shapes") for l in s])
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
    # sample yields, identified by "rate"
    idx = next(
        i for i, s in enumerate(sections_list) if any(["rate" in l[0:4] for l in s])
    )
    sections_dict.update({"channels": sections_list.pop(idx)})
    
    # UPDATED: systematics, last in list (need better identifier)
    # in the shapes configuration, last lines can be of format:
    # <something> autoMCStats <many something>
    # <something> rateParam <many something>
    # we exclude them and try to understand what they do later on
    modifiers = []
    unsupported_modifiers = ["autoMCStats", "rateParam", "param"]
    for i, line in enumerate(sections_list[-1]):
        if any(unsupported_modifier in line for unsupported_modifier in unsupported_modifiers):
            break
        else:
            modifiers.append(sections_list[-1].pop(i))
    sections_dict.update({"modifiers": modifiers})

    # full list of channels and samples from datacard (including duplications)
    channel_names = sections_dict["channels"][0].split()[1:]
    channel_yields = [float(y) for y in sections_dict["channels"][3].split()[1:]]
    sample_names = sections_dict["channels"][1].split()[1:]
    sample_numbers = sections_dict["channels"][2].split()[1:]
    modifier_names = [line.split()[0] for line in sections_dict["modifiers"]]
    #log.debug("Modifiers: {}".format(modifier_names))

    # restructure shapes
    mass = "800"
    if "shapes" in sections_dict:
        sections_dict["shapes"] = restructure_shapes(sections_dict["shapes"], channel_names, sample_names, modifier_names, mass)
    else:
        sections_dict["shapes"] = []

    # convert observations into dict
    sections_dict["observations"] = restructure_observations(
        sections_dict["observations"], sections_dict["shapes"]
    )

    # convert channel information (sample yields) into dict
    sections_dict["channels"] = restructure_channels(sections_dict["channels"])

    # convert modifier information into dict
    # needs access to full lists of channels (+ yields) and sample names
    sections_dict["modifiers"] = restructure_modifiers(
        sections_dict["modifiers"], channel_names, channel_yields, sample_names, sample_numbers, sections_dict["shapes"]
    )

    #log.debug("sections_dict = {}".format(sections_dict))

    return sections_dict


def extract_histogram(file: str, histo: str) -> dict:
    #log.debug("Extracting {} from {}".format(histo, file))
    f = uproot.open(file)
    shape = {}
    histo = f[histo]
    shape["values"] = list(histo.values())
    shape["edges"] = list(histo.axis().edges())

    return shape


def sections_dict_to_workspace(sections_dict: dict) -> dict:
    """convert dictionary with info from datacard into workspace"""
    ws = {}
    # need to add signal POI manually, assuming signal is first process
    for channel in sections_dict["channels"]:
        for i, sample in enumerate(channel["samples"]):
            # attach modifiers to this sample for this channel
            sample["modifiers"] = sections_dict["modifiers"][channel["name"]][
                sample["name"]
            ]
            # attach normfactor for signals
            # signals are identified by an id <= 0
            idn = int(sample["id"])
            if idn <= 0:
                # this should be signal, attach normfactor
                sample["modifiers"].append(
                    {"data": None, "name": "r_{}".format(sample["name"]), "type": "normfactor"}
                )
            # time to add shapes, if shapes analysis
            if sections_dict["shapes"]:
                # loop to check which one matches all the requirements
                for shape_dict in sections_dict["shapes"]:
                    if "modifier" not in shape_dict and shape_dict["sample"] == sample["name"] \
                        and shape_dict["channel"] == channel["name"]:
                        sample["shape"] = extract_histogram(shape_dict["file"], shape_dict["histogram"])
                        break

    ws.update({"channels": sections_dict["channels"]})
    ws.update(
        {"measurements": [{"config": {"parameters": [], "poi": "r"}, "name": "meas"}]}
    )
    ws.update({"observations": sections_dict["observations"]})
    ws.update({"version": "1.0.0"})

    return ws


def datacard_to_json(datacard: list) -> dict:
    sections_dict = get_sections_dict(datacard)
    ws = sections_dict_to_workspace(sections_dict)
    #log.debug(f"HistFactory workspace in JSON:\n{json_str(ws)}\n")
    return ws


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
    
    with open(sys.argv[-1]) as f:
        datacard = f.readlines()

    ws = datacard_to_json(datacard)
    ws_name = ".".join(sys.argv[-1].split(".")[0:-1]) + ".json"

    log.info(f"saving workspace as {ws_name}")
    with open(ws_name, "w") as f:
        f.write(json_str(ws))
