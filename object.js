const response = {
  contentString: "",
  toolCalls: [
    {
      name: "Search",
      args: {
        __arg1: "株式会社Elith 住所",
      },
      id: "call_UC4HRikkr2hd3F5hkeSWzG1J",
      type: "tool_call",
    },
  ],
  content: "",
  additional_kwargs: {
    tool_calls: [
      {
        id: "call_UC4HRikkr2hd3F5hkeSWzG1J",
        function: {
          arguments: '{"__arg1": "株式会社Elith 住所"}',
          name: "Search",
        },
        type: "function",
      },
    ],
    refusal: null,
  },
  response_metadata: {
    token_usage: {
      completion_tokens: 19,
      prompt_tokens: 91,
      total_tokens: 110,
      completion_tokens_details: {
        accepted_prediction_tokens: 0,
        audio_tokens: 0,
        reasoning_tokens: 0,
        rejected_prediction_tokens: 0,
      },
      prompt_tokens_details: {
        audio_tokens: 0,
        cached_tokens: 0,
      },
    },
    model_name: "gpt-4o-mini-2024-07-18",
    system_fingerprint: "fp_fff1d3b4b6",
    finish_reason: "tool_calls",
    logprobs: null,
  },
  id: "run-f970181f-074e-46bb-820f-ffcc49395c0e-0",
  tool_calls: [
    {
      name: "Search",
      args: {
        __arg1: "株式会社Elith 住所",
      },
      id: "call_UC4HRikkr2hd3F5hkeSWzG1J",
      type: "tool_call",
    },
  ],
  usage_metadata: {
    input_tokens: 91,
    output_tokens: 19,
    total_tokens: 110,
  },
};

const searchResult = [
  "株式会社 Elith type: Computer consultant in Bunkyō, Japan.",
  "株式会社 Elith entity_type: local_nav.",
  "株式会社 Elith kgmid: /g/11x7yhkxfd.",
  "株式会社 Elith place_id: ChIJlSbgVo73Fg0RhBVMyBn_maA.",
  "株式会社 Elith address: Japan, 〒113-0033 Tokyo, Bunkyo City, Hongo, 2 Chome−27−17 Frontier Hongo 旧 JP 113-0033ICNビル) 6階A室.",
  "株式会社 Elith phone: +81 80 - 9759 - 8070.",
  "株式会社 Elith raw_hours: Open · Closes 6 PM.",
  '株式会社Elith merchant_description: "株式会社Elithは、LLM時代の「速く・安全に・確実に成果を出す」実装を支援する日本発のAIスタートアップです。 個社特化のソリューションを提供する受託開発に加えて、幻覚・コンプライアンス・説明可能性を定量化するリスクマネジメントツール「GENFLUX」と、社員のAI人材化・AIに関する経営判断を支援する人材育成・教育事業も行っています。".',
  "会社情報.\u200b株式会社Elith ; 所在地. 113-0033 東京都文京区本郷2 - 27 - 17 フロンティア本郷 6階 A室 ; 代表取締役.\u200b井上 顧基 ; \u200b設立.\u200b2022年12月 ; 事業内容.AI（人工知能） ...",
  "株式会社Elith✓.elith_pr.Location: 東京都文京区本郷.Website: https://www.",
  "本田技研工業株式会社. 16 ◇ お問い合わせ 会社名 株式会社Elith 住所 〒113-0033 東京都文京区本郷3-30-10本郷 K&Kビル1F 電話番号. 03-6822-5999 ...",
  "会社情報. 株式会社Elith. 東京都文京区本郷2-27-17 フロンティア本郷Ⅰ 6-A. https://www.elith.ai. 2022/12 に設立. 井上 顧基 が創業. 9人のメンバー. 類似の会社.",
  "株式会社Elithは、クライアントと共に課題を発見し、AIによる最適な解決策を共創するパートナーです。製造業、金融業、医療業など、さまざまな業種のクライアントの ...",
  "株式会社Elithについて. 社名：株式会社Elith. 代表者：代表取締役 井上 顧基. 所在地：東京都文京区本郷 3-30-10 本郷 K&K ビル 1F. 設立：2022 年 12 月.",
  "企業名: 株式会社Elith ; 英語名: Elith Inc. ; 代表者名: 井上 顧基 ; 住所: 東京都文京区本郷3-30-10 本郷K&Kビル 1F ; URL: https://www.elith.ai/.",
  "採用企業情報. 株式会社Elith. 東京都. 会社規模非公開. 会社概要. 【設立】2022年12月【代表者】井上 顧基【本社所在地】東京都文京区本郷2-27-17",
  "株式会社Elith（エリス）は、2022年設立の東京都渋谷区恵比寿西2丁目3番13号に所在する法人です（法人番号: 6011001151296）。最終登記更新は2022/12/07 ...",
];
