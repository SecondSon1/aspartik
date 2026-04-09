import subprocess
import tempfile
from collections.abc import Sequence
from typing import Literal, Optional

from aspartik.data.msa import MSA

from ._shared import CalculatorKind, SubstitutionModel, TreePrior


def _format_taxa(
    names: list[str],
    heights: Optional[Sequence] = None,
) -> str:
    if heights:
        taxa = [
            f'<taxon id="{name}">\n\t\t<date value="{height}" direction="backwards" units="years"/>\n\t</taxon>'
            for name, height in zip(names, heights)
        ]
    else:
        taxa = [f'<taxon id="{name}"/>' for name in names]
    return "\n\t\t".join(taxa)


def _format_sequences(msa: MSA) -> str:
    sequences = []
    for i in range(msa.num_sequences):
        name = msa.sequence_name(i)
        seq = str(msa.sequence(i))
        sequences.append(
            f'<sequence>\n\t\t\t<taxon idref="{name}"/>\n\t\t\t{seq}\n\t\t</sequence>'
        )
    return "\n\t\t".join(sequences)


_beast1_default_template = """<?xml version="1.0" standalone="yes"?>

<beast version="10.5.0">
    <taxa id="taxa">
        {taxa}
    </taxa>

    <alignment id="alignment" dataType="nucleotide">
        {sequences}
    </alignment>

    <patterns id="patterns" from="1" strip="false">
        <alignment idref="alignment"/>
    </patterns>

{tree_init}

{tree_prior}

{clock}

{substitution_model}

    <siteModel id="siteModel">
        <substitutionModel>
            <HKYModel idref="hky"/>
        </substitutionModel>
    </siteModel>

{likelihood_block}

    <operators id="operators" optimizationSchedule="log">
{operators}
    </operators>

    <mcmc id="mcmc" chainLength="{length}" autoOptimize="false" adaptation="false">
        <posterior id="posterior">
            <prior id="prior">
                {priors}

                <strictClockBranchRates idref="branchRates"/>
            </prior>
            <likelihood id="likelihood">
                {likelihood_ref}
            </likelihood>
        </posterior>
        <operators idref="operators"/>

        {screen_log}

        {file_log}

        {tree_log}
    </mcmc>

    <report>
        <property name="timer">
            <mcmc idref="mcmc"/>
        </property>
    </report>
</beast>
"""


def _file_log(log: str, log_path: Optional[str]) -> str:
    if not log_path:
        return ""

    return f"""
            <log id="fileLog" logEvery="1000" fileName="{log_path}">
                <posterior idref="posterior"/>
                <prior idref="prior"/>
                <likelihood idref="likelihood"/>
                <treeHeightStatistic id="tree:height">
                    <treeModel idref="tree"/>
                </treeHeightStatistic>
                <treeLengthStatistic id="tree:length">
                    <treeModel idref="tree"/>
                </treeLengthStatistic>

    {log}
            </log>
    """


def _screen_log(screen_log_every: Optional[int]) -> str:
    if not screen_log_every:
        return ""

    return f"""
        <log id="screenLog" logEvery="{screen_log_every}">
            <column label="Posterior" dp="4" width="12">
                <posterior idref="posterior"/>
            </column>
            <column label="Prior" dp="4" width="12">
                <prior idref="prior"/>
            </column>
            <column label="Likelihood" dp="4" width="12">
                <likelihood idref="likelihood"/>
            </column>
        </log>
    """


def _tree_log(tree_log_path: Optional[str], tree_log_every: int) -> str:
    if not tree_log_path:
        return ""

    return f"""
        <logTree id="treeFileLog" logEvery="{tree_log_every}" fileName="{tree_log_path}">
            <treeModel idref="tree"/>
        </logTree>
"""


def beast1_config(
    msa: MSA,
    *,
    heights: Optional[Sequence] = None,
    newick: Optional[str] = None,
    substitution_model: SubstitutionModel,
    kappa: float = 2.0,
    frequencies: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
    operator_mix: Literal["default", "classic"] = "default",
    clock_rate: Optional[float] = None,
    tree_prior: Optional[TreePrior] = None,
    use_beagle: bool = True,
    log_path: Optional[str] = None,
    tree_log_path: Optional[str] = None,
    tree_log_every: int = 1_000,
    screen_log_every: Optional[int] = 1_000,
    length: int,
):
    operators, priors, log = "", "", ""

    taxa = _format_taxa(list(msa.sequence_names()), heights)
    sequences = _format_sequences(msa)

    if newick:
        tree_init = f"""\
    <newick id="startingTree" usingDates="false">
        {newick}
    </newick>

    <treeModel id="tree">
        <newick idref="startingTree"/>
        <rootHeight>
            <parameter id="tree.rootHeight"/>
        </rootHeight>
        <nodeHeights internalNodes="true">
            <parameter id="tree.internalNodeHeights"/>
        </nodeHeights>
        <nodeHeights internalNodes="true" rootNode="true">
            <parameter id="tree.allInternalNodeHeights"/>
        </nodeHeights>
    </treeModel>"""
    else:
        tree_init = """\
    <constantSize id="_starting_coalescent" units="years">
        <populationSize>
            <parameter id="_starting_population_size" value="100.0" lower="0.0"/>
        </populationSize>
    </constantSize>
    <coalescentSimulator id="startingTree">
        <taxa idref="taxa"/>
        <constantSize idref="_starting_coalescent"/>
    </coalescentSimulator>

    <treeModel id="tree">
        <coalescentTree idref="startingTree"/>
        <rootHeight>
            <parameter id="tree.rootHeight"/>
        </rootHeight>
        <nodeHeights internalNodes="true">
            <parameter id="tree.internalNodeHeights"/>
        </nodeHeights>
        <nodeHeights internalNodes="true" rootNode="true">
            <parameter id="tree.allInternalNodeHeights"/>
        </nodeHeights>
    </treeModel>"""

    freq_str = " ".join(str(f) for f in frequencies)

    substitution_model_s = None
    match substitution_model:
        case "HKY":
            substitution_model_s = f"""
    <HKYModel id="hky">
        <frequencies>
            <frequencyModel dataType="nucleotide">
                <frequencies>
                    <parameter id="frequencies" value="{freq_str}"/>
                </frequencies>
            </frequencyModel>
        </frequencies>
        <kappa>
            <parameter id="kappa" value="{kappa}" lower="0.0"/>
        </kappa>
    </HKYModel>
"""
            operators += """
        <scaleOperator scaleFactor="0.75" weight="1">
            <parameter idref="kappa"/>
        </scaleOperator>
        <deltaExchange delta="0.01" weight="3">
            <parameter idref="frequencies"/>
        </deltaExchange>
            """

            priors += """
                <logNormalPrior id="prior.kappa" mu="1.0" sigma="1.25" offset="0.0">
                    <parameter idref="kappa"/>
                </logNormalPrior>
            """

            log += """
            <parameter idref="kappa"/>
            <parameter idref="frequencies"/>
            """

    if newick:
        operator_mix = "classic"

    match operator_mix:
        case "default":
            num = min(msa.num_sequences, 1000)
            operators += f"""
        <upDownOperator scaleFactor="0.75" weight="3">
            <up>
                <parameter idref="tree.allInternalNodeHeights"/>
            </up>
            <down>
                <parameter idref="clock_rate"/>
            </down>
        </upDownOperator>
        <subtreeLeap size="1.0" weight="{num}">
            <treeModel idref="tree"/>
        </subtreeLeap>
        <fixedHeightSubtreePruneRegraft weight="{num / 10}">
            <treeModel idref="tree"/>
        </fixedHeightSubtreePruneRegraft>
            """

    clock_s = None
    match clock_rate:
        case None:
            operators += """
        <scaleOperator scaleFactor="0.75" weight="3">
            <parameter idref="clock_rate"/>
        </scaleOperator>
            """

            priors += """
                <laplacePrior id="prior:clock_rate" mean="0.0" scale="0.5">
                    <parameter idref="clock_rate"/>
                </laplacePrior>
            """

            clock_s = """
    <strictClockBranchRates id="branchRates">
        <rate>
            <parameter id="clock_rate" value="1.0" lower="0.0"/>
        </rate>
    </strictClockBranchRates>
            """

            log += """
            <parameter idref="clock_rate"/>
            """

        case float(clock_rate):
            clock_s = f"""
    <strictClockBranchRates id="branchRates">
        <rate>
            <parameter id="clock_rate" value="{clock_rate}" lower="0.0"/>
        </rate>
    </strictClockBranchRates>
            """

    assert clock_s is not None

    tree_prior_s = ""
    match tree_prior:
        case None:
            pass
        case "constant":
            tree_prior_s = """
    <constantSize id="constant_population" units="years">
        <populationSize>
            <parameter id="population_size" value="1.0" lower="0.0"/>
        </populationSize>
    </constantSize>
    <coalescentLikelihood id="prior:coalescent">
        <model>
            <constantSize idref="constant_population"/>
        </model>
        <intervals>
            <treeIntervals>
                <treeModel idref="tree"/>
            </treeIntervals>
        </intervals>
    </coalescentLikelihood>
            """

            operators += """
        <scaleOperator scaleFactor="0.75" weight="3">
            <parameter idref="population_size"/>
        </scaleOperator>
            """

            priors += """
                <gammaPrior id="prior:population_size" shape="0.001" scale="1000.0" offset="0.0">
                    <parameter idref="population_size"/>
                </gammaPrior>

                <coalescentLikelihood idref="prior:coalescent"/>
            """

            log += """
            <parameter idref="population_size"/>
            <coalescentLikelihood idref="prior:coalescent"/>
            """

        case "exponential":
            tree_prior_s = """
    <exponentialGrowth id="exponential_growth" units="years">
        <populationSize>
            <parameter id="population_size" value="1.0" lower="0.0" />
        </populationSize>
        <growthRate>
            <parameter id="growth_rate" value="1.0" />
        </growthRate>
    </exponentialGrowth>
    <coalescentLikelihood id="prior:coalescent">
        <model>
            <exponentialGrowth idref="exponential_growth"/>
        </model>
        <intervals>
            <treeIntervals>
                <treeModel idref="tree"/>
            </treeIntervals>
        </intervals>
    </coalescentLikelihood>
            """

            operators += """
        <scaleOperator scaleFactor="0.75" weight="3">
            <parameter idref="population_size"/>
        </scaleOperator>
		<randomWalkOperator windowSize="1.0" weight="3">
			<parameter idref="growth_rate"/>
		</randomWalkOperator>
            """

            priors += """
                <gammaPrior id="prior:population_size" shape="0.001" scale="1000.0" offset="0.0">
                    <parameter idref="population_size"/>
                </gammaPrior>
				<laplacePrior mean="0" scale="100">
					<parameter idref="growth_rate"/>
				</laplacePrior>

                <coalescentLikelihood idref="prior:coalescent"/>
            """

            log += """
            <parameter idref="population_size"/>
            <parameter idref="growth_rate"/>
            <coalescentLikelihood idref="prior:coalescent"/>
            """

        case "yule":
            tree_prior_s = """
    <yulemodel id="yule" units="years">
        <birthRate>
            <parameter id="birth_rate" value="2.0" lower="0.0"/>
        </birthRate>
    </yulemodel>
    <speciationLikelihood id="prior:yule">
        <model>
            <yuleModel idref="yule"/>
        </model>
        <speciesTree>
            <treeModel idref="tree"/>
        </speciesTree>
    </speciationLikelihood>
            """

            operators += """
        <scaleOperator scaleFactor="0.75" weight="3">
            <parameter idref="birth_rate"/>
        </scaleOperator>
            """

            priors += """
                <logNormalPrior mu="1.0" sigma="1.5" offset="0.0">
                    <parameter idref="birth_rate"/>
                </logNormalPrior>
                <speciationLikelihood idref="prior:yule"/>
            """

            log += """
            <parameter idref="birth_rate"/>
            <speciationLikelihood idref="prior:yule"/>
            """

    if use_beagle:
        likelihood_block = """\
    <treeDataLikelihood id="treeLikelihood" useAmbiguities="false" usePreOrder="false">
        <partition>
            <patterns idref="patterns"/>
            <siteModel idref="siteModel"/>
        </partition>
        <treeModel idref="tree"/>
        <strictClockBranchRates idref="branchRates"/>
    </treeDataLikelihood>"""
        likelihood_ref = '<treeDataLikelihood idref="treeLikelihood"/>'
    else:
        likelihood_block = """\
    <treeLikelihood id="treeLikelihood" useAmbiguities="false">
        <patterns idref="patterns"/>
        <treeModel idref="tree"/>
        <siteModel idref="siteModel"/>
        <strictClockBranchRates idref="branchRates"/>
    </treeLikelihood>"""
        likelihood_ref = '<treeLikelihood idref="treeLikelihood"/>'

    return _beast1_default_template.format(
        taxa=taxa,
        sequences=sequences,
        tree_init=tree_init,
        likelihood_block=likelihood_block,
        likelihood_ref=likelihood_ref,
        clock=clock_s,
        substitution_model=substitution_model_s,
        operators=operators,
        tree_prior=tree_prior_s,
        priors=priors,
        file_log=_file_log(log, log_path),
        screen_log=_screen_log(screen_log_every),
        tree_log=_tree_log(tree_log_path, tree_log_every),
        length=length,
    )


def beast1_run(
    config: str,
    calculator: CalculatorKind = "cpu",
    overwrite: bool = True,
    seed: int = 4,
):
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w+t") as tmp:
        tmp.write(config)
        tmp.flush()

        args = ["beast", "-seed", str(seed), "-citations_off"]
        if overwrite:
            args.append("-overwrite")
        match calculator:
            case "cpu":
                args.append("-beagle_CPU")
            case "cuda":
                args.append("-beagle_cuda")
        args.append(tmp.name)

        subprocess.run(args)


def beast1_likelihood(
    msa: MSA,
    newick: str,
    kappa: float,
    frequencies: tuple[float, ...],
    clock_rate: float,
) -> float:
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = f"{tmpdir}/beast.log"
        config = beast1_config(
            msa,
            substitution_model="HKY",
            newick=newick,
            kappa=kappa,
            frequencies=frequencies,
            clock_rate=clock_rate,
            use_beagle=False,
            log_path=log_path,
            screen_log_every=None,
            length=0,
        )

        xml_path = f"{tmpdir}/beast.xml"
        with open(xml_path, "w") as f:
            f.write(config)

        result = subprocess.run(
            ["beast", "-seed", "1", "-citations_off", "-overwrite", xml_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"BEAST1 failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        df = pd.read_csv(log_path, sep="\t", comment="#")
        return float(df["likelihood"].iloc[0])
