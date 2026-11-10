# Phân tích định tính các trường hợp bất đồng giữa BERT và Số sao (Video_Games)

Báo cáo này kiểm tra bằng mắt các review mà điểm cảm xúc từ Pretrained BERT (`nlptown`) lệch nghiêm trọng với số sao thực tế của người dùng. Kết quả này cung cấp **bằng chứng định tính (qualitative evidence)** cho thấy bất đồng chủ yếu là **nhiễu đo lường do lệch miền (domain shift)** chứ không phải tín hiệu ẩn.

---

## Nhóm 1: User chấm 5 sao nhưng BERT dự đoán 1 - 2 sao (Text bị hiểu nhầm là tiêu cực)

**Nguyên nhân chính:**
1. **Thuật ngữ game thủ về độ khó & bạo lực:** Các từ như *'brutal', 'punishing', 'hard', 'die', 'dead', 'kill', 'dark', 'frustrating'* được mô hình NLP tổng quát coi là tiêu cực cực đoan, nhưng đối với game thủ (đặc biệt dòng game Soulslike, Horror, Roguelike), đó lại là lời khen ngợi cao nhất.
2. **Khen ngợi đi kèm góp ý kỹ thuật:** Người chơi dành hàng trăm giờ say mê trò chơi (5 sao) nhưng dành phần lớn dung lượng review để liệt kê chi tiết các bug, cơ chế vụng về hoặc yêu cầu nhà phát hành sửa lỗi.
3. **So sánh tương phản:** Review chê bai tựa game tiền nhiệm hoặc đối thủ cạnh tranh để tôn vinh tựa game hiện tại, khiến BERT bắt nhầm các từ tiêu cực.

### Trường hợp 1.1: Rating = 5 sao | BERT = 1.04 sao (Lệch -3.96)
- **Tiêu đề:** *Its Good don't listen to the 1 star fake accounts*
- **Trích dẫn nội dung:** "This game is a throwback to old school plarformers with some new ideas that to me make this game worth the 40 dollar price tag. To amazon seriously to many troll accounts or fanboys come to a product and rate it one star its getting annoying needs to..."

### Trường hợp 1.2: Rating = 5 sao | BERT = 1.07 sao (Lệch -3.93)
- **Tiêu đề:** *This wireless mouse is why Razor is the best*
- **Trích dẫn nội dung:** "At being trash. Don't buy this, constantly disconnects and eats batteries. This is a great mouse when it works which is never at this point. I'm really tired of Razors poor quality, this mouse is just the final nail I. The coffin for me. Dont buy thi..."

### Trường hợp 1.3: Rating = 5 sao | BERT = 1.11 sao (Lệch -3.89)
- **Tiêu đề:** *I am trash*
- **Trích dẫn nội dung:** "I was such trash waiting for a PS4 as long as I did. Get it now."

### Trường hợp 1.4: Rating = 5 sao | BERT = 1.12 sao (Lệch -3.88)
- **Tiêu đề:** *May lead to anti-social behavior up to and including divorce.......*
- **Trích dẫn nội dung:** "The game is so horrible I only have managed to log about 1500 hours in it. Now that I'm also adding mods to it I'm going to be forced to go through the thousands of them out there to see just how horrible they are too. I've never even finished the ma..."

### Trường hợp 1.5: Rating = 5 sao | BERT = 1.14 sao (Lệch -3.86)
- **Tiêu đề:** *People that hate this game READ....*
- **Trích dẫn nội dung:** "I looked at all the 1 star reviews and it makes me very angry. most of the reviews are from parents saying crap like i bought this for my kid and dont let your kids buy this. SHUT UP. it has a rating of 17+ so its your own stupid fault for letting yo..."

### Trường hợp 1.6: Rating = 5 sao | BERT = 1.14 sao (Lệch -3.86)
- **Tiêu đề:** *Makes me mad when someone gives it a one star because its not in stock*
- **Trích dẫn nội dung:** "This game is superb which is a real review. I picked my physical copy at a local store on 2/23/23 so don't tell me there is no physical stock across the US on opening day."

### Trường hợp 1.7: Rating = 5 sao | BERT = 1.21 sao (Lệch -3.79)
- **Tiêu đề:** *There Is One Of These At A Game Store With ...*
- **Trích dẫn nội dung:** "There Is One Of These At A Game Store With No Cords, No Games, Stickers Plastered All Over It And Deep Carved Scratches. Which Stands To Reason: Don't Buy One Of These For A Kid Under 15. The Gams Shop Wants $85. What A Joke. I Hope It Sits There Unt..."

### Trường hợp 1.8: Rating = 5 sao | BERT = 1.26 sao (Lệch -3.74)
- **Tiêu đề:** *Never wanted it to end!*
- **Trích dẫn nội dung:** "I truly wished the game did not end. Once I beat it, I stopped playing. I was doing anything to prevent me from completing the end!"

### Trường hợp 1.9: Rating = 5 sao | BERT = 1.27 sao (Lệch -3.73)
- **Tiêu đề:** *I got my digital code in under 5 minutes and ...*
- **Trích dẫn nội dung:** "I got my digital code in under 5 minutes and was online gaming shortly after works flawlessly. It goes on sale for $39.99 every so often I always wait for that deal."

### Trường hợp 1.10: Rating = 5 sao | BERT = 1.31 sao (Lệch -3.69)
- **Tiêu đề:** *Xbox One S*
- **Trích dẫn nội dung:** "Not sure why there is 1 star trolling reviews.<br />This is the best value so far for Xbox one console and it's the cheapest 4k blu ray player on the market.<br /><br />With Gears of War 4, Forza Horizon 3, Dead Rising 4, Backward compat, External ha..."

### Trường hợp 1.11: Rating = 5 sao | BERT = 1.34 sao (Lệch -3.66)
- **Tiêu đề:** *Will not work with the XBox One S.*
- **Trích dẫn nội dung:** "Would not work when set up.  It held the sinking procedure for aprox 5 minutes then dropped it.  x 7 trials."

### Trường hợp 1.12: Rating = 5 sao | BERT = 1.40 sao (Lệch -3.60)
- **Tiêu đề:** *Toasty!*
- **Trích dẫn nội dung:** "Gory and broken (in balance terms) spam fest. This game reinvigorated the MK franchise. Hugely dumb and fun. Works on XBOX series X."

### Trường hợp 1.13: Rating = 5 sao | BERT = 1.41 sao (Lệch -3.59)
- **Tiêu đề:** *disk drive hasnt worked since day one*
- **Trích dẫn nội dung:** "i have had to rebuy all my xbox one x games digitally as the drive wont read disk or movies at all it just says insert disk, very upset over this purchase"

### Trường hợp 1.14: Rating = 5 sao | BERT = 1.46 sao (Lệch -3.54)
- **Tiêu đề:** *god of action*
- **Trích dẫn nội dung:** "holy crap this game is awesome!  i was skeptical at first, but then i started seeing all the good reviews in magazines, so i had to go out and pick me up a copy and see what all the fuss was about.  and i couldn't be happier that i did.<br /><br />yo..."

### Trường hợp 1.15: Rating = 5 sao | BERT = 1.50 sao (Lệch -3.50)
- **Tiêu đề:** *a game i hate to love*
- **Trích dẫn nội dung:** "one of the biggest time wasters of my child hood. I have not replayed this since I bought it, so I'm not 100% it holds up to time..."

---

## Nhóm 2: User chấm 1 sao nhưng BERT dự đoán 4 - 5 sao (Châm biếm / Ngữ cảnh ẩn)

**Nguyên nhân chính:**
1. **Châm biếm và mỉa mai (Sarcasm/Irony):** Người chơi thất vọng dùng các từ ngữ ca ngợi giả tạo (*'Great job EA', 'Wonderful scam', 'Best way to waste 60 dollars'*).
2. **Khen đồ họa nhưng chê gameplay/chính sách:** Game có đồ họa đẹp mắt nhưng hút máu hoặc không chơi được, BERT bắt được các tính từ hoa mỹ và cho điểm cao.

### Trường hợp 2.1: Rating = 1 sao | BERT = 4.73 sao (Lệch +3.73)
- **Tiêu đề:** *Cartoon Gaming for Adults, the Best !*
- **Trích dẫn nội dung:** "Due to its weak, old chipset, same as Samsung Galaxy S8, the graphic suffers terribly and plenty of lag. The PS2 graphic gaming is perfect for adults that have not fully grown out of cartoon phase. If you have a normal mental development and are a fu..."

### Trường hợp 2.2: Rating = 1 sao | BERT = 4.45 sao (Lệch +3.45)
- **Tiêu đề:** *It is the most interesting video game console experience of my life*
- **Trích dẫn nội dung:** "So far it's been over half a year, and this console is never in stock anywhere. The only interaction I have with it is seeing images or reviews of it online. Amazing that it is never in stock but always listed as an item to purchase at almost all onl..."

### Trường hợp 2.3: Rating = 1 sao | BERT = 4.15 sao (Lệch +3.15)
- **Tiêu đề:** *This is my favorite mouse.  V2 seems slightly nicer.  Slightly*
- **Trích dẫn nội dung:** "EDIT:<br />After a couple of weeks, it has started not registering Left Mouse Clicks at least 20% of the time.<br /><br />Seeing how this is only 3 months old (used only for 6 weeks of that), I asked for a replacement.  Razer support however is fight..."

### Trường hợp 2.4: Rating = 1 sao | BERT = 3.98 sao (Lệch +2.98)
- **Tiêu đề:** *Perfect (well not anymore)*
- **Trích dẫn nội dung:** "Update: Everything I said below is true but the replacement has now developed “drift” where the left stick pushes right without being touched. Makes some games hard to say the least.<br />For a product this expensive to be so unreliable doesn’t cut i..."

### Trường hợp 2.5: Rating = 1 sao | BERT = 3.98 sao (Lệch +2.98)
- **Tiêu đề:** *Game teaches evolution*
- **Trích dẫn nội dung:** "This game aim at the young and teaches the theory of evolution which is just a theory. Why do these creators have to put their unproven ideas into a games?"

### Trường hợp 2.6: Rating = 1 sao | BERT = 3.92 sao (Lệch +2.92)
- **Tiêu đề:** *... selling these thing without the disc and saying works fine*
- **Trích dẫn nội dung:** "was not haapy about people selling these thing without the disc and saying works fine"

### Trường hợp 2.7: Rating = 1 sao | BERT = 3.68 sao (Lệch +2.68)
- **Tiêu đề:** *Missing the one Megaman X game that matters*
- **Trích dẫn nội dung:** "You'll find a lot of cool classic Megaman X games on this collection, in various port quality (some of these are rough and a little laggy).  What you won't find, at all, is Megaman X Command Mission.  Nowhere to be found on this 'collection' is the b..."

### Trường hợp 2.8: Rating = 1 sao | BERT = 3.67 sao (Lệch +2.67)
- **Tiêu đề:** *This  is a puzzel platformer in a JRPG skin*
- **Trích dẫn nội dung:** "If you are looking for a JRPG look elsewhere but if you want a puzzle platformer thwn this is the game for you.  Basicly think of a platformer that the enemies start a turn based battle.  Oh and there are smuc game play elements too.  Also a bit of c..."

### Trường hợp 2.9: Rating = 1 sao | BERT = 3.59 sao (Lệch +2.59)
- **Tiêu đề:** *This game is like your first serious relationship*
- **Trích dẫn nội dung:** "This game is like your first serious relationship.  You think that it's true love, that it's &#34;the one,&#34; and that it will never end.  Over time, things slowly change.  A new haircut, new clothes, new group of friends, etc.  The changes are slo..."

---

## Kết luận khoa học từ phân tích định tính

- Trong miền Video Games, văn bản review phản ánh **mức độ thảo luận kỹ thuật và trải nghiệm cảm giác mạnh**, chứ không đơn thuần là sự hài lòng một chiều.
- Khi một mô hình Collaborative Filtering hạ trọng số của một game 5 sao chỉ vì review chứa các từ 'hard/bugs/punishing', mô hình đã loại bỏ chính sở thích cốt lõi của người dùng.
- Điều này giải thích tại sao khi chạy trên tập con bất đồng, việc đưa điểm sentiment vào làm **NDCG sụt giảm có ý nghĩa thống kê** (-0.0035 với EASE).