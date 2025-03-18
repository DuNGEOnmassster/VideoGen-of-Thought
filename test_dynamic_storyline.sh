#!/bin/bash

# Type 1: Character Life Stories (5 examples)
mkdir -p asset/story_type1
python scripts/dynamic_storyline_system.py \
    --user_input "a story of a classic American woman Mary's life, from birth to death." \
    --story_name "Mary" \
    --num_shot 30

python scripts/dynamic_storyline_system.py \
    --user_input "the journey of a boy named Alex who grows up in a small fishing village and becomes a renowned marine biologist." \
    --story_name "Alex" \
    --num_shot 25

python scripts/dynamic_storyline_system.py \
    --user_input "an immigrant's story of moving to a new country, struggling, and eventually finding success as an entrepreneur." \
    --story_name "Immigrant_Journey" \
    --num_shot 20

python scripts/dynamic_storyline_system.py \
    --user_input "a talented musician's rise to fame, struggles with addiction, and eventual redemption." \
    --story_name "Musician_Story" \
    --num_shot 30

python scripts/dynamic_storyline_system.py \
    --user_input "the life of an ordinary office worker who discovers extraordinary abilities after a freak accident and uses them to help others." \
    --story_name "Hidden_Hero" \
    --num_shot 25

# Type 2: Multi-character Adventures (5 examples)
mkdir -p asset/story_type2
python scripts/dynamic_storyline_system.py \
    --user_input "a group of four friends embarking on a treasure hunt that tests their friendship and courage." \
    --story_name "Treasure_Hunters" \
    --num_shot 20

python scripts/dynamic_storyline_system.py \
    --user_input "a team of scientists on an expedition to discover a lost civilization in the Amazon rainforest." \
    --story_name "Lost_Expedition" \
    --num_shot 25

python scripts/dynamic_storyline_system.py \
    --user_input "a space mission gone wrong, where the crew must work together to survive and return to Earth." \
    --story_name "Space_Survival" \
    --num_shot 20

python scripts/dynamic_storyline_system.py \
    --user_input "a detective and a reluctant civilian witness solving a mysterious series of art thefts." \
    --story_name "Art_Mystery" \
    --num_shot 15

python scripts/dynamic_storyline_system.py \
    --user_input "a family road trip across America that brings them closer together through unexpected adventures." \
    --story_name "Family_Journey" \
    --num_shot 20

# Type 3: Flexible Narratives (5 examples)
mkdir -p asset/story_type3
python scripts/dynamic_storyline_system.py \
    --user_input "a week-long food festival in a small Italian town, showcasing different cuisines, chefs, and cultural exchanges." \
    --story_name "Food_Festival" \
    --num_shot 15

python scripts/dynamic_storyline_system.py \
    --user_input "a day in a bustling city market from dawn to midnight, following different vendors and customers." \
    --story_name "Market_Day" \
    --num_shot 12

python scripts/dynamic_storyline_system.py \
    --user_input "the renovation of an old theater and the stories of the people involved in bringing it back to life." \
    --story_name "Theater_Revival" \
    --num_shot 18

python scripts/dynamic_storyline_system.py \
    --user_input "a three-day music festival from the perspectives of performers, organizers, and attendees." \
    --story_name "Music_Festival" \
    --num_shot 15

python scripts/dynamic_storyline_system.py \
    --user_input "a series of connected moments in a small town as it prepares for and experiences a once-in-a-century solar eclipse." \
    --story_name "Eclipse_Day" \
    --num_shot 20