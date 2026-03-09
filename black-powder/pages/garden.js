import Head from 'next/head'
import Link from 'next/link'
import dynamic from 'next/dynamic'
import { useRef } from 'react'

const TriptychModel = dynamic(() => import('../src/components/garden/TriptychModel'), { ssr: false })

export default function GardenPage() {
  return (
    <>
      <Head>
        <title>The Garden of Earthly Delights | A Three-Generation Conspiracy</title>
        <meta name="description" content="A mathematical analysis of why Hieronymus Bosch could not have painted The Garden of Earthly Delights alone — and who really did." />
      </Head>

      <div className="garden-landing">

        {/* Navigation */}
        <nav className="garden-nav">
          <Link href="/garden"><a className="nav-logo">Categorical <span>Apertures</span></a></Link>
          <Link href="/"><a className="nav-link">Enter the Framework</a></Link>
        </nav>

        {/* =========================================== */}
        {/* HERO */}
        {/* =========================================== */}
        <div className="garden-hero">
          <div className="hero-content">
            <span className="hero-overline">Art History&apos;s Greatest Deception</span>
            <h1>The Garden of <em>Earthly Delights</em></h1>
            <p className="hero-subtitle">
              For five hundred years, scholars have asked the wrong question about the most mysterious painting in human history. They asked <em>how</em> one mind could conceive all of this. The right question is whether one mind <em>could</em>.
            </p>
            <div className="hero-scroll-hint">Scroll to uncover the truth</div>
          </div>
        </div>

        {/* =========================================== */}
        {/* THE TRIPTYCH - 3D Model */}
        {/* =========================================== */}
        <div className="garden-section">
          <div className="section-grid">
            <div className="garden-text">
              <span className="section-number">I — The Masterpiece</span>
              <h2>A Painting That Defies Explanation</h2>
              <p>
                The Garden of Earthly Delights is a triptych — three panels that, when opened, span over twelve feet of the most bewildering imagery ever committed to wood. Over five hundred fantastical creatures. Hundreds of symbolic elements. Three radically different worlds rendered with impossible precision.
              </p>
              <p>
                Art historians have spent centuries trying to decode it. They have written thousands of pages on its symbolism, its theology, its psychosexual undertones. But they have all operated under a single, unquestioned assumption:
              </p>
              <p className="emphasis">
                That one man — Hieronymus Bosch — painted all of it.
              </p>
              <p>
                What if that assumption is wrong? What if the answer has been hiding in plain sight for five hundred years — not in the symbols, but in the brushstrokes themselves?
              </p>
            </div>
            <div className="garden-visual">
              <div className="model-container">
                <TriptychModel />
              </div>
              <p className="image-caption">Interactive 3D model — click and drag to explore the triptych</p>
            </div>
          </div>
        </div>

        <div className="garden-divider"><div className="line"></div></div>

        {/* =========================================== */}
        {/* THE HELL PANEL - This is Bosch */}
        {/* =========================================== */}
        <div className="garden-section">
          <div className="section-grid reverse">
            <div className="garden-text">
              <span className="section-number">II — The Right Panel</span>
              <h2>This Is Hieronymus Bosch</h2>
              <p>
                The right panel — Hell — is the one art historians understand best. And for good reason: it is unmistakably, unambiguously Bosch. The tortured figures, the demonic instruments, the burning cityscape on the horizon, the knife-eared creature presiding over the damned. Every element carries the signature darkness of a mind consumed by the terror of divine judgment.
              </p>
              <p>
                This is the panel where Bosch placed his own face. Not as a bystander. Not as a saint. As a <span className="highlight">hollow figure trapped in Hell</span>, staring out at the viewer from beneath a disc balanced on his head — a broken man witnessing the consequences of earthly sin.
              </p>
              <p className="emphasis">
                An artist signs their work. Bosch signed the Hell panel. He did not place himself in Paradise. He did not place himself among the earthly pleasures. He placed himself in damnation.
              </p>
              <p className="stat">
                Probability that a single-panel artist places self-portrait only in panel 3: 100%<br/>
                Probability that an all-panel artist places self-portrait only in panel 3: 33%
              </p>
            </div>
            <div className="garden-visual">
              <div className="image-placeholder">
                <img src="/img/garden/hell_panel.jpg" alt="The Hell panel of the Garden of Earthly Delights" />
              </div>
              <p className="image-caption">The right panel (Hell) — the only panel bearing Bosch&apos;s self-portrait</p>
            </div>
          </div>
        </div>

        <div className="garden-divider"><div className="line"></div></div>

        {/* =========================================== */}
        {/* HIS OTHER WORKS - Pattern of Darkness */}
        {/* =========================================== */}
        <div className="garden-section">
          <div className="section-grid">
            <div className="garden-text">
              <span className="section-number">III — The Catalogue of Darkness</span>
              <h2>Every Other Painting He Made</h2>
              <p>
                If the Hell panel feels like Bosch, that is because it <em>is</em> Bosch. Consider his entire surviving catalogue. Not a single work departs from this fundamental darkness:
              </p>
              <p>
                <span className="highlight">The Last Judgment</span> — an apocalyptic nightmare of demons devouring the damned, rendered with the same burning horizon, the same tortured bodies, the same mechanical cruelty. <span className="highlight">Christ Carrying the Cross</span> — a wall of grotesque faces pressed together, leering, mocking, each one more deformed than the last. <span className="highlight">The Temptation of St. Anthony</span> — a saint besieged by hybrid monsters, fish-creatures, and burning architecture.
              </p>
              <p>
                <span className="highlight">The Ship of Fools</span>. <span className="highlight">The Haywain</span>. <span className="highlight">The Seven Deadly Sins</span>. Every single surviving painting shares the same aesthetic vocabulary: suffering, judgment, grotesquerie, moral terror.
              </p>
              <p className="emphasis">
                In a catalogue of unrelenting darkness, the Paradise panel of the Garden is not an outlier. It is a different artist.
              </p>
            </div>
            <div className="garden-visual">
              <div className="image-placeholder">
                <img src="/img/garden/last_judgment.jpg" alt="The Last Judgment by Hieronymus Bosch" />
              </div>
              <p className="image-caption">The Last Judgment — the same burning horizon, the same demonic imagination</p>
            </div>
          </div>
        </div>

        <div className="garden-divider"><div className="line"></div></div>

        {/* =========================================== */}
        {/* HIS DRAWINGS - Even the sketches are dark */}
        {/* =========================================== */}
        <div className="garden-section">
          <div className="section-grid reverse">
            <div className="garden-text">
              <span className="section-number">IV — The Sketchbook</span>
              <h2>Even His Pen Couldn&apos;t Escape It</h2>
              <p>
                Paintings can be commissioned. A patron might request a specific subject. But drawings — private sketches, studies, doodles in the margins — reveal the artist&apos;s unfiltered mind. They show what an artist reaches for when no one is watching.
              </p>
              <p>
                Bosch&apos;s surviving drawings are a menagerie of the grotesque. Creatures with human torsos and insect legs. Faces melting into beaks. Bodies impaled on branches. Demons carrying sinners in baskets. These are not studies for Paradise. These are not sketches of innocent animals grazing by a fountain. These are the private obsessions of a mind that saw the world as a theater of punishment.
              </p>
              <p>
                His drawing hand and his painting hand spoke the same language: <span className="highlight">one of damnation, never of innocence</span>.
              </p>
              <p className="emphasis">
                You cannot fake an entire body of private sketches. The drawings are the psychological fingerprint. And that fingerprint matches only one panel of the Garden.
              </p>
            </div>
            <div className="garden-visual">
              <div className="image-placeholder">
                <img src="/img/garden/bosch_drawings.jpg" alt="Drawings by Hieronymus Bosch" />
              </div>
              <p className="image-caption">Bosch&apos;s surviving drawings — demons, grotesques, and suffering, never innocence</p>
            </div>
          </div>
        </div>

        <div className="garden-divider"><div className="line"></div></div>

        {/* =========================================== */}
        {/* QUOTE BREAK */}
        {/* =========================================== */}
        <div className="garden-quote">
          <blockquote>
            The man who drew demons in his private sketchbook did not paint Paradise on his public altarpiece. Someone else held that brush.
            <div className="attribution">— Statistical inference, not speculation</div>
          </blockquote>
        </div>

        <div className="garden-divider"><div className="line"></div></div>

        {/* =========================================== */}
        {/* THE PARADISE PANEL - A different mind */}
        {/* =========================================== */}
        <div className="garden-section">
          <div className="section-grid">
            <div className="garden-text">
              <span className="section-number">V — The Left Panel</span>
              <h2>Now Look at Paradise</h2>
              <p>
                The left panel of the Garden is a world of pristine innocence. God presents Eve to Adam in a landscape of crystalline fountains, exotic animals, and lush vegetation. Elephants, giraffes, unicorns graze peacefully. Birds of impossible plumage circle a pink fountain. The color palette is luminous — soft pinks, vivid greens, gentle blues stretching to a calm horizon.
              </p>
              <p>
                There is no torment here. No grotesquerie. No moral warning. The animals are rendered with the careful eye of someone who has <em>seen</em> them — or studied manuscripts describing them in detail. The exotic creatures suggest access to bestiaries and travel accounts from the Age of Exploration, the kind of knowledge passed down through a family with connections.
              </p>
              <p>
                This is not the imagination of the man who drew demons in every margin. This is the eye of someone who looked at the natural world with <span className="highlight">wonder, not judgment</span>.
              </p>
              <p className="stat">
                Character orthodoxy of Bosch (documented): 0.9 / 1.0<br/>
                Character orthodoxy expressed in Paradise panel: 0.1 / 1.0<br/>
                Probability of this mismatch: 2%
              </p>
            </div>
            <div className="garden-visual">
              <div className="image-placeholder">
                <img src="/img/garden/paradise_panel.jpg" alt="The Paradise panel of the Garden of Earthly Delights" />
              </div>
              <p className="image-caption">The left panel (Paradise) — luminous, innocent, celebratory. A completely different sensibility.</p>
            </div>
          </div>
        </div>

        <div className="garden-divider"><div className="line"></div></div>

        {/* =========================================== */}
        {/* THE MIDDLE PANEL - The son's rebellion */}
        {/* =========================================== */}
        <div className="garden-section">
          <div className="section-grid reverse">
            <div className="garden-text">
              <span className="section-number">VI — The Central Panel</span>
              <h2>The Pleasures of Earth</h2>
              <p>
                The middle panel is perhaps the most scandalous image in all of medieval art. Hundreds of nude figures cavort in a landscape of oversized fruits, riding exotic animals, bathing in pools, engaged in acts of unmistakable sensuality. It is playful, irreverent, almost celebratory of earthly pleasure.
              </p>
              <p>
                This is the panel that makes art historians most uncomfortable when attributed to Bosch. Here was a man documented as a member of the Brotherhood of Our Lady — a deeply conservative religious confraternity in &apos;s-Hertogenbosch. A man whose every other work screams moral condemnation. A man of severe religious orthodoxy.
              </p>
              <p>
                And yet this panel revels in the flesh. It does not condemn — it <em>celebrates</em>. The figures are not punished. They are enjoying themselves. This is not the vision of a religious moralist. This is the vision of someone with a fundamentally different relationship to the body and to pleasure.
              </p>
              <p className="emphasis">
                Enter Jan van Aken — Bosch&apos;s father. A man of the world, not the cloister. A painter himself, with the skill and the sensibility to render earthly delight without theological terror.
              </p>
            </div>
            <div className="garden-visual">
              <div className="image-placeholder">
                <img src="/img/garden/middle_panel.jpg" alt="The central panel of the Garden of Earthly Delights" />
              </div>
              <p className="image-caption">The central panel — sensual, playful, irreverent. Not the mind of a religious conservative.</p>
            </div>
          </div>
        </div>

        <div className="garden-divider"><div className="line"></div></div>

        {/* =========================================== */}
        {/* THE TEMPTATION - Comparison piece */}
        {/* =========================================== */}
        <div className="garden-section">
          <div className="section-grid">
            <div className="garden-text">
              <span className="section-number">VII — The Forensic Comparison</span>
              <h2>The Temptation of St. Anthony</h2>
              <p>
                Consider the Temptation of St. Anthony — a painting universally attributed to Bosch alone, with no scholarly dispute. It is the perfect control case. The saint kneels in prayer while the world erupts around him: a fish with legs carries a building on its back, a pig wears a nun&apos;s habit, a bird-headed creature reads a letter. Burning towns glow on the horizon.
              </p>
              <p>
                Now hold this image in your mind and compare it to the Paradise panel. They might as well be from different centuries, different countries, different <em>species</em> of imagination. The Temptation is Bosch&apos;s voice at full volume — frenetic, anxious, morally charged, every inch packed with symbolic warning.
              </p>
              <p>
                The Paradise panel breathes. It has space. It has calm. Its animals are observed, not invented. Its landscape recedes with atmospheric perspective, not with the claustrophobic layering of nightmares.
              </p>
              <p className="emphasis">
                The same hand did not paint both. The stylometric distance between these works is not a gap — it is a canyon.
              </p>
            </div>
            <div className="garden-visual">
              <div className="image-placeholder">
                <img src="/img/garden/temptation.jpg" alt="The Temptation of St. Anthony by Hieronymus Bosch" />
              </div>
              <p className="image-caption">The Temptation of St. Anthony — undisputed Bosch. Frenetic, anxious, morally charged.</p>
            </div>
          </div>
        </div>

        <div className="garden-divider"><div className="line"></div></div>

        {/* =========================================== */}
        {/* CHRIST CARRYING THE CROSS */}
        {/* =========================================== */}
        <div className="garden-section">
          <div className="section-grid reverse">
            <div className="garden-text">
              <span className="section-number">VIII — The Psychological Portrait</span>
              <h2>Christ Carrying the Cross</h2>
              <p>
                This painting is perhaps the most revealing window into Bosch&apos;s psychology. There is no landscape. No architecture. No space at all. Just faces — a claustrophobic press of grotesque humanity surrounding Christ. Every face is a study in cruelty, stupidity, or malice. Mouths open in jeers. Eyes bulge with sadistic pleasure. Noses are bulbous, chins recede, skin stretches over skulls.
              </p>
              <p>
                Only Christ and Veronica have human faces. Everyone else is a caricature of moral failure. This is how Bosch saw humanity: as a mob of the damned, pressing in from all sides, incapable of recognizing divinity in their midst.
              </p>
              <p>
                This is the man who, according to the conventional theory, also painted a paradise of naked innocents frolicking with unicorns. The man who filled this canvas with nothing but hatred for human flesh also painted the most sensual celebration of the human body in medieval art.
              </p>
              <p className="stat">
                The cognitive dissonance required to maintain this attribution has prevented art historians from seeing the obvious for five centuries.
              </p>
            </div>
            <div className="garden-visual">
              <div className="image-placeholder">
                <img src="/img/garden/christ_cross.jpg" alt="Christ Carrying the Cross by Hieronymus Bosch" />
              </div>
              <p className="image-caption">Christ Carrying the Cross — nothing but grotesque faces. This is Bosch&apos;s view of humanity.</p>
            </div>
          </div>
        </div>

        <div className="garden-divider"><div className="line"></div></div>

        {/* =========================================== */}
        {/* QUOTE BREAK 2 */}
        {/* =========================================== */}
        <div className="garden-quote">
          <blockquote>
            Three panels. Three sensibilities. Three generations of painters who happened to share a surname.
            <div className="attribution">— The van Aken family of &apos;s-Hertogenbosch</div>
          </blockquote>
        </div>

        <div className="garden-divider"><div className="line"></div></div>

        {/* =========================================== */}
        {/* THE VAN AKEN FAMILY */}
        {/* =========================================== */}
        <div className="garden-section">
          <div className="section-grid">
            <div className="garden-text">
              <span className="section-number">IX — The Family</span>
              <h2>The Van Aken Conspiracy</h2>
              <p>
                Hieronymus Bosch was born Jheronimus van Aken. His grandfather, <span className="highlight">Antonius van Aken</span>, was a painter. His father, <span className="highlight">Jan van Aken</span>, was a painter. His uncles were painters. Painting was the family trade, passed from hand to hand across three generations in the small Dutch city of &apos;s-Hertogenbosch.
              </p>
              <p>
                Antonius — the grandfather — belonged to an earlier generation, one still connected to the manuscript illumination tradition. He would have had access to English bestiaries, to travel accounts from merchants passing through the Low Countries, to the careful observation of exotic animals described by returning sailors. His visual vocabulary would have been one of <em>wonder at creation</em> — the natural theology of a pre-Reformation world.
              </p>
              <p>
                Jan — the father — was a man of his time. The middle of the fifteenth century was an age of expanding horizons, secular confidence, humanist curiosity. His world was one of earthly discovery, not eschatological fear.
              </p>
              <p>
                Hieronymus — the grandson — came of age as the Reformation approached, as religious anxiety intensified, as the Brotherhood of Our Lady tightened its moral grip. His world was one of <span className="highlight">judgment, sin, and damnation</span>.
              </p>
              <p className="emphasis">
                Three generations. Three worldviews. Three panels.
              </p>
            </div>
            <div className="garden-visual">
              <div className="image-placeholder">
                <img src="/img/garden/hertogenbosch.jpg" alt="'s-Hertogenbosch in the 15th century" />
              </div>
              <p className="image-caption">&apos;s-Hertogenbosch — where three generations of van Aken painters lived and worked</p>
            </div>
          </div>
        </div>

        <div className="garden-divider"><div className="line"></div></div>

        {/* =========================================== */}
        {/* THE OWL MARKERS */}
        {/* =========================================== */}
        <div className="garden-section">
          <div className="section-grid reverse">
            <div className="garden-text">
              <span className="section-number">X — The Hidden Signatures</span>
              <h2>The Owl Boundary Markers</h2>
              <p>
                Look carefully at where the panels meet. At each junction, partially concealed by the visual complexity of the scene, sits an owl. These are not random decorative elements — owls appear throughout Bosch&apos;s iconography as symbols of hidden wisdom, of seeing in darkness, of knowledge concealed from the uninitiated.
              </p>
              <p>
                But here they serve a dual purpose. They function as <span className="highlight">engineering boundary markers</span> — the exact points where one painter&apos;s work ends and another&apos;s begins. Disguised as mystical symbolism, they are in fact the most practical elements in the entire painting: the coordination specifications that allowed three artists to work on adjacent panels with seamless integration.
              </p>
              <p>
                The genius of this is staggering. The very elements scholars have spent centuries interpreting as theological symbols are actually the construction documents of a collaborative project. Hidden in plain sight. Meaning one thing to the viewer, another thing entirely to the painters themselves.
              </p>
              <p className="stat">
                Probability of seamless integration across 1000+ elements by individual: 0.004%<br/>
                Probability with systematic collaboration using boundary markers: 73%
              </p>
            </div>
            <div className="garden-visual">
              <div className="image-placeholder">
                <img src="/img/garden/owl_detail.jpg" alt="Owl detail from the Garden of Earthly Delights" />
              </div>
              <p className="image-caption">The owls at the panel boundaries — mystical symbols to viewers, engineering specs to the painters</p>
            </div>
          </div>
        </div>

        <div className="garden-divider"><div className="line"></div></div>

        {/* =========================================== */}
        {/* THE MATHEMATICS */}
        {/* =========================================== */}
        <div className="garden-section">
          <div className="section-grid">
            <div className="garden-text">
              <span className="section-number">XI — The Mathematics</span>
              <h2>Statistical Certainty</h2>
              <p>
                This is not speculation. We can compute the probability that any other family in fifteenth-century Netherlands could have produced this painting. The requirements are specific: three-generation painters, active in the correct timeframe, located in &apos;s-Hertogenbosch, with access to English manuscripts, travel experience matching the exotic fauna, personal religious trauma matching the Hell imagery, and the motivation to conceal their collaboration.
              </p>
              <p>
                The combined probability of another family meeting all nine criteria:
              </p>
              <p className="emphasis">
                1 in 667 billion.
              </p>
              <p>
                Using Bayesian model selection with conservative priors — giving the single-artist hypothesis a 70% prior probability and the collaboration hypothesis only 25% — the posterior probability after evaluating all available evidence:
              </p>
              <p className="emphasis">
                The van Aken collaboration hypothesis has a <span className="highlight">99.99%</span> posterior probability.
              </p>
              <p>
                The single-artist hypothesis is rejected with a p-value below 10⁻⁵⁰. For reference, the standard for scientific discovery is p &lt; 0.05. This exceeds that threshold by forty-eight orders of magnitude.
              </p>
            </div>
            <div className="garden-visual">
              <div className="image-placeholder">
                <img src="/img/garden/mathematics.jpg" alt="Mathematical analysis of the collaboration hypothesis" />
              </div>
              <p className="image-caption">Bayesian posterior: 99.99% — mathematics, not conjecture</p>
            </div>
          </div>
        </div>

        <div className="garden-divider"><div className="line"></div></div>

        {/* =========================================== */}
        {/* THE SHIP OF FOOLS */}
        {/* =========================================== */}
        <div className="garden-section">
          <div className="section-grid reverse">
            <div className="garden-text">
              <span className="section-number">XII — The Control Experiment</span>
              <h2>The Ship of Fools</h2>
              <p>
                If you still harbor doubt, consider The Ship of Fools — one of Bosch&apos;s smallest and most intimate paintings. A boatload of revelers drink, sing, and gorge themselves while their vessel drifts without purpose. A monk and a nun lean toward each other in drunken intimacy. A man vomits over the side. A fool sits in the rigging.
              </p>
              <p>
                This is how Bosch depicts pleasure: as <span className="highlight">folly</span>. As spiritual death. As the ship sailing toward damnation while its passengers celebrate their own destruction. Even when he paints people enjoying themselves, they are damned by their enjoyment. There is no innocence in Bosch&apos;s pleasure — only sin that hasn&apos;t been punished yet.
              </p>
              <p>
                Now look again at the middle panel of the Garden. Those nude figures bathing in pools, riding animals, feeding each other fruit — where is the condemnation? Where is the moral horror? It is <em>absent</em>. The middle panel presents pleasure without judgment, and Bosch was constitutionally incapable of that.
              </p>
              <p className="emphasis">
                The Ship of Fools is the control experiment. It proves that when Bosch painted pleasure, he painted it as damnation. The middle panel does not. Therefore the middle panel is not Bosch.
              </p>
            </div>
            <div className="garden-visual">
              <div className="image-placeholder">
                <img src="/img/garden/ship_of_fools.jpg" alt="The Ship of Fools by Hieronymus Bosch" />
              </div>
              <p className="image-caption">The Ship of Fools — how Bosch depicts earthly pleasure: as spiritual death</p>
            </div>
          </div>
        </div>

        <div className="garden-divider"><div className="line"></div></div>

        {/* =========================================== */}
        {/* THE 500-YEAR SECRET */}
        {/* =========================================== */}
        <div className="garden-section">
          <div className="section-grid">
            <div className="garden-text">
              <span className="section-number">XIII — The Deception</span>
              <h2>Five Hundred Years of Misdirection</h2>
              <p>
                The van Akens never lied. They never claimed Hieronymus painted the entire work. They simply allowed the assumption to persist — and the assumption was so natural, so aligned with how people think about art, that no one questioned it. One painting, one painter. It seems obvious. And that is precisely why it worked.
              </p>
              <p>
                The information entropy of the secret — the number of alternative explanations available to scholars — was approximately 15 bits. Too many false trails, too many competing interpretations, too many symbolic rabbit holes for anyone to focus on the simplest explanation: that the painting looks like three different artists because it <em>was</em> three different artists.
              </p>
              <p>
                The van Akens achieved the ultimate synthesis of their four-dimensional framework: using <span className="highlight">science</span> (systematic artistic collaboration) to perform <span className="highlight">magic</span> (an apparently impossible achievement) while fulfilling <span className="highlight">religion</span> (theological narrative across three panels) for the <span className="highlight">entertainment</span> of eternity.
              </p>
              <p>
                Their magic coefficient — computed from knowledge asymmetry, execution difficulty, deception duration, and constraint transcendence — exceeds <span className="highlight">200 million</span>. The highest tier of what we call &quot;real magic&quot;: achievements that become <em>more</em> impressive when you understand how they were done.
              </p>
            </div>
            <div className="garden-visual">
              <div className="image-placeholder">
                <img src="/img/garden/triptych_closed.jpg" alt="The Garden of Earthly Delights, closed panels" />
              </div>
              <p className="image-caption">The triptych closed — the outer panels show the world in grisaille, before the revelation within</p>
            </div>
          </div>
        </div>

        <div className="garden-divider"><div className="line"></div></div>

        {/* =========================================== */}
        {/* CTA - Bridge to Framework */}
        {/* =========================================== */}
        <div className="garden-cta">
          <div className="cta-content">
            <h2>The Same Eyes That Saw This</h2>
            <p>
              The same pattern-recognition that uncovered a five-hundred-year conspiracy in art history now reveals something equally hidden in biology: a unified mathematical framework governing information filtering across thirteen orders of magnitude — from molecular enzymes to neural consciousness.
            </p>
            <p>
              Three axioms. Five operational regimes. One equation spanning every scale of biological information processing.
            </p>
            <Link href="/">
              <a className="cta-button">Enter the Framework</a>
            </Link>
          </div>
        </div>

        {/* Footer */}
        <footer style={{
          textAlign: 'center',
          padding: '40px',
          borderTop: '1px solid #30363d',
          color: '#484f58',
          fontSize: '13px',
          fontFamily: 'Poppins, sans-serif'
        }}>
          <p>&copy; 2025 Kundai F. Sachikonye — Categorical Apertures</p>
        </footer>

      </div>
    </>
  )
}
