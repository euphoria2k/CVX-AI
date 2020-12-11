import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router);

export default new Router({
    scrollBehavior() {
        return window.scrollTo({ top: 0, behavior: 'smooth' });
    },
    routes: [
        {
            path: '/VideoPlayer',
            name: 'videoPlayer',
            component: () => import('../Pages/VideoPlayer.vue'),
        },
        {
            path: '/',
            name: 'HomePage',
            component: () => import('../Pages/HomePage.vue'),
        },
    ]
})
