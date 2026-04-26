// Optional Supabase Edge Function (admin only)
// Reads a JSON payload of story cards and calls the import_cards RPC.
//
// Deploy with service-role secrets only. Do not expose this to Unity clients.

import { createClient } from "npm:@supabase/supabase-js@2";

Deno.serve(async (req) => {
  try {
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const serviceRoleKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, serviceRoleKey);

    const body = await req.json();
    const cards = body.cards;
    const deckVersion = body.deck_version ?? "v1";

    const { data, error } = await supabase.rpc("import_cards", {
      p_cards: cards,
      p_deck_version: deckVersion,
    });

    if (error) {
      return new Response(JSON.stringify({ status: "error", error }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response(JSON.stringify(data), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    return new Response(JSON.stringify({ status: "error", message: String(err) }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
});
