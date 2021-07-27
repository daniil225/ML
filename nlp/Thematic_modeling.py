from nltk.tokenize import TreebankWordTokenizer
from collections import Counter

kite_text  = """A kite is a flying tethered object that depends upon the tension of a tethering system. The necessary lift that makes the kite wing fly is generated when air (or in some cases water) flows over and under the kite's wing, producing low pressure above the wing and high pressure below it. This deflection also generates horizontal drag along the direction of the wind. The resultant force vector from the lift and drag force components is opposed by the tension of the one or more lines or tethers. The anchor point of the kite line may be static or moving (e.g., the towing of a kite by a running person, boat, or vehicle).

Kites may be flown for recreation, art or other practical uses. Sport kites can be flown in aerial ballet, sometimes as part of a competition. Power kites are multi-line steerable kites designed to generate large forces which can be used to power activities such as kite surfing, kite landboarding or kite buggying. Kites towed behind boats can lift passengers which has had useful military applications in the past."""

kite_history = """The kite was first invented and popularized approximately 2,800 years ago in China, where materials ideal for kite building were readily available: silk fabric for sail material, fine, high-tensile-strength silk for flying line, and resilient bamboo for a strong, lightweight framework. Alternatively, kite author Clive Hart and kite expert Tal Streeter hold that kites existed far before that time. The kite was said to be the invention of the famous 5th century BC Chinese philosophers Mozi and Lu Ban. By at least 549 AD paper kites were being flown, as it was recorded in that year a paper kite was used as a message for a rescue mission. Ancient and medieval Chinese sources list other uses of kites for measuring distances, testing the wind, lifting men, signalling, and communication for military operations. The earliest known Chinese kites were flat (not bowed) and often rectangular. Later, tailless kites incorporated a stabilizing bowline. Kites were decorated with mythological motifs and legendary figures; some were fitted with strings and whistles to make musical sounds while flying.[12]

After its appearance in China, the kite migrated to Japan, Korea, Thailand, Burma (Myanmar), India, Arabia, and North Africa, then farther south into the Malay Peninsula, Indonesia, and the islands of Oceania as far east as Easter Island. Since kites made of leaves have been flown in Malaya and the South Seas from time immemorial, the kite could also have been invented independently in that region."""


tokenizer = TreebankWordTokenizer()

kite_intro = kite_text.lower()
intro_tokens = tokenizer.tokenize(kite_intro)
kite_history = kite_history.lower()
history_tokens = tokenizer.tokenize(kite_history)

intro_total = len(intro_tokens)
history_total = len(history_tokens)


intro_tf = {}
history_tf = {}
intro_counters = Counter(intro_tokens)
intro_tf['kite'] = intro_counters['kite'] / intro_total

history_counts = Counter(history_tokens)
history_tf['kite'] = history_counts['kite'] / history_total


