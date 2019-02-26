#  Copyright 2016,2017 Eric S. Tellez <eric.tellez@infotec.mx>
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


export Sss, LaesaTournament #, KvpTournament

function Sss(db::Array{T,1}, dist::D, alpha::Float64, shuffle_db::Bool=false) where {T,D}
    pivots = [db[refID] for refID in select_sss(db, dist, alpha, shuffle_db)]
    fit(Laesa, db, dist, pivots)
end

function LaesaTournament(db::Array{T,1}, dist::D, numrefs::Int, tournamentsize::Int=3) where {T,D}
    pivots = [db[refID] for refID in select_tournament(db, dist, numrefs, tournamentsize)]
    fit(Laesa, db, dist, pivots)
end

# function KvpTournament(db::Array{T,1}, dist::D, k::Int, numrefs::Int, tournamentsize::Int=0) where {T,D}
#     #pivots = [db[refID] for refID in select_tournament(db, numrefs, tournamentsize, dist)]
#     pivots = select_tournament(db, dist, numrefs, tournamentsize)
#     Kvp(db, dist, k, pivots)
# end
